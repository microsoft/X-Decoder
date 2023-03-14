import sys
import os
import re
import math
import glob
import uuid
import torch
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import cv2
import einops
from scipy import ndimage
from pytorch_lightning import seed_everything
import random
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from torchvision import transforms
from utils.arguments import load_opt_from_config_file
from utils.distributed import init_distributed
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
metedata = MetadataCatalog.get('coco_2017_train_panoptic')

X_CHAT_PREFIX = """X-Chat is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. X-Chat is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

X-Chat is able to process and understand large amounts of text and images. As a language model, X-Chat can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and X-Chat can invoke different tools to indirectly understand pictures. When talking about images, X-Chat is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, X-Chat is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. X-Chat is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to X-Chat with a description. The description helps X-Chat to understand this image, but X-Chat should use tools to finish following tasks, rather than directly imagine from the description.

Overall, X-Chat is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

X-Chat has access to the following tools:"""

X_CHAT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

X_CHAT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since X-Chat is a text language model, X-Chat must use tools to observe images rather than imagination.
The thoughts and observations are only visible for X-Chat, X-Chat should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"hitory_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    else:
        paragraphs = history_memory.split('\n')
        last_n_tokens = n_tokens
        while last_n_tokens >= keep_last_n_words:
            last_n_tokens = last_n_tokens - len(paragraphs[0].split(' '))
            paragraphs = paragraphs[1:]
        return '\n' + '\n'.join(paragraphs)

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[0:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
        recent_prev_file_name = name_split[0]
        new_file_name = '{}_{}_{}_{}.png'.format(this_new_uuid, func_name, recent_prev_file_name, most_org_file_name)
    return os.path.join(head, new_file_name)

def crop_image(input_image):
    h, w = input_image.shape[1:]
    crop_h, crop_w = math.floor(h/64) * 64, math.floor(w/64) * 64
    im_cropped = input_image[:, :crop_h, :crop_w]
    return im_cropped

class XDecoder: 
    def __init__(self, device):
        self.device = device
        opt = load_opt_from_config_file("configs/xdecoder/svlp_focalt_lang.yaml")
        opt = init_distributed(opt)
        self.model = BaseModel(opt, build_model(opt)).from_pretrained("xdecoder_focalt_last_novg.pt").eval().to(device)
        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(["background", "background"], is_eval=True)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt",).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(image.size)

xdmodel = XDecoder(device="cuda")

class ImageEditing:
    def __init__(self, device):
        print("Initializing StableDiffusionInpaint to %s" % device)
        self.device = device
        self.processor = transforms.Compose(
            [
                transforms.Resize(512, interpolation=Image.BICUBIC), 
                transforms.PILToTensor()
            ]
        )
        self.inpainting = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",revision="fp16",torch_dtype=torch.float16,).to(device)

    def remove_part_of_image(self, input):
        image_path, to_be_removed_txt = input.split(",")
        print(f'remove_part_of_image: to_be_removed {to_be_removed_txt}')
        return self.replace_part_of_image(f"{image_path},{to_be_removed_txt},clean and empty scene")

    def replace_part_of_image(self, input):
        image_path, to_be_replaced_txt, replace_with_txt = input.split(",")
        print(f'replace_part_of_image: replace_with_txt {replace_with_txt}')
        original_image = Image.open(image_path)
        cropped_image = crop_image(self.processor(original_image)).to(self.device)
        texts = [[texts if to_be_replaced_txt.strip().endswith('.') else (to_be_replaced_txt.strip() + '.')]]
        batch_inputs = [{'image': cropped_image, 'height': cropped_image.shape[1], 'width': cropped_image.shape[2], 'groundings': {'texts': texts}}]        
        outputs = xdmodel.model.model.evaluate_grounding(batch_inputs, None)
        grd_mask = (outputs[0]['grounding_mask'] > 0).float().cpu().numpy()        

        # dialate the mask a little bit
        struct2 = ndimage.generate_binary_structure(2, 2)
        mask_dilated = ndimage.binary_dilation(grd_mask[0], structure=struct2, iterations=3).astype(grd_mask[0].dtype)
        mask_image = Image.fromarray(mask_dilated * 255).convert('RGB')
        width, height = mask_image.size[0], mask_image.size[1]
        updated_image = self.inpainting(prompt=replace_with_txt, image=transforms.ToPILImage()(cropped_image).convert('RGB'), mask_image=mask_image, height=height, width=width).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image.save(updated_image_path)
        return updated_image_path

class Pix2Pix:
    def __init__(self, device):
        print("Initializing Pix2Pix to %s" % device)
        self.device = device
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting Pix2Pix Inference")
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = self.pipe(instruct_text,image=original_image,num_inference_steps=40,image_guidance_scale=1.2,).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        return updated_image_path

class T2I:
    def __init__(self, device):
        print("Initializing T2I to %s" % device)
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        self.text_refine_tokenizer = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_model = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.text_refine_gpt2_pipe = pipeline("text-generation", model=self.text_refine_model, tokenizer=self.text_refine_tokenizer, device=self.device)
        self.pipe.to(device)

    def inference(self, text):
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        refined_text = text # self.text_refine_gpt2_pipe(text)[0]["generated_text"]
        print(f'{text} refined to {refined_text}')
        image = self.pipe(refined_text).images[0]
        image.save(image_filename)
        print(f"Processed T2I.run, text: {text}, image_filename: {image_filename}")
        return image_filename

class ImageCaptioning:
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.processor = transforms.Compose([transforms.Resize(224, interpolation=Image.BICUBIC)])

    def inference(self, image_path):
        image = self.processor(Image.open(image_path).convert("RGB"))
        width = image.size[0]; height = image.size[1]        
        image = np.asarray(image)
        images = torch.from_numpy(image.copy()).permute(2,0,1).to(self.device)
        batch_inputs = [{'image': images, 'height': height, 'width': width, 'image_id': 0}]
        outputs = xdmodel.model.model.evaluate_captioning(batch_inputs)
        captions = outputs[-1]['captioning_text']
        return captions

class ImageRetrieval:
    def __init__(self, device):
        print("Initializing ImageRetrieval to %s" % device)
        self.device = device
        self.processor = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC), 
                transforms.PILToTensor()
            ]
        )
        self.v_emb = None
        self.image_folder_path = "image_pool"
        self._get_image_list()

    def _get_image_list(self):
        imgs_root = self.image_folder_path
        self.img_pths = sorted(
            glob.glob(os.path.join(imgs_root, '*.jpg')) + glob.glob(os.path.join(imgs_root, '*.png')) + glob.glob(os.path.join(imgs_root, '*.webp'))
        )
        if len(self.img_pths) == 0:
            return
        
        if len(self.img_pths) > 100:
            self.img_pths = self.img_pths[:100]

        self.imgs = [Image.open(x).convert('RGB') for x in self.img_pths]
        images = [self.processor(x).to(self.device) for x in self.imgs]
        batch_inputs = [{'image': image, 'image_id': 0} for image in images]
        outputs = xdmodel.model.model.evaluate(batch_inputs)
        v_emb = torch.cat([x['captions'][-1:] for x in outputs])
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        self.v_emb = v_emb

    def inference(self, text_query):
        if self.v_emb is None:
            print("Please upload images to image_pool folder")
            return ""

        texts = text_query
        texts_ = [[x.strip() if x.strip().endswith('.') else (x.strip() + '.')] for x in texts.split(',')][0]
        xdmodel.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(texts_, is_eval=True, name='caption', prompt=False)
        t_emb = getattr(xdmodel.model.model.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('caption'))
        # temperature = xdmodel.model.model.sem_seg_head.predictor.lang_encoder.logit_scale       

        logits = torch.matmul(self.v_emb, t_emb.t())
        prob, idx = logits.max(0)
        if prob.view(-1).item() > 0.2:
            new_img_path = self.img_pths[idx].replace('image_pool', 'image')
            head, tail = os.path.split(new_img_path)
            new_image_name = tail.replace(tail[:tail.find('.')], str(uuid.uuid4())[0:4])
            new_img_path = os.path.join(head, new_image_name)
            updated_image_path = get_new_image_name(new_img_path, func_name="imgret")
            self.imgs[idx].save(updated_image_path)
            return updated_image_path, "similarity is {}".format(prob.view(-1).item())
        else:
            return "the image is not good, need to generate an image instead"

class ReferringImageCaptioning:
    def __init__(self, device):
        print("Initializing ReferringImageCaptioning to %s" % device)
        self.device = device
        self.processor = transforms.Compose([transforms.Resize(224, interpolation=Image.BICUBIC)])

    def inference(self, input):
        image_path, target_object = input.split(",")
        image = self.processor(Image.open(image_path).convert("RGB"))
        image = np.asarray(image)
        images = torch.from_numpy(image.copy()).permute(2,0,1).to(self.device)
        texts_input = [[target_object.strip() if target_object.endswith('.') else (target_object + '.')]]
        batch_inputs = [{'image': images, 'groundings': {'texts':texts_input}, 'height': images.shape[1], 'width': images.shape[2]}]
        # grounding first
        outputs = xdmodel.model.model.evaluate_grounding(batch_inputs, None)
        grd_mask_ = outputs[-1]['grounding_mask']
        color = [252/255, 91/255, 129/255]
        visual = Visualizer(image, metadata=metedata)
        demo = visual.draw_binary_mask((grd_mask_>0).float().cpu().numpy()[0], color=color, text=target_object)
        res = demo.get_image()

        # captioning then
        token_text = target_object.replace('.','') if target_object.endswith('.') else target_object
        token = xdmodel.model.model.sem_seg_head.predictor.lang_encoder.tokenizer.encode(token_text)
        token = torch.tensor(token)[None,:-1]

        batch_inputs = [{'image': images, 'image_id': 0, 'captioning_mask': grd_mask_}]
        outputs = xdmodel.model.model.evaluate_captioning(batch_inputs, extra={'token': token})
        captions = outputs[-1]['captioning_text']
        return captions

class ReferringImageSegmentation:
    def __init__(self, device):
        print("Initializing ReferringImageSegmentation to %s" % device)
        self.device = device
        self.processor = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC)])

    def inference(self, input):
        image_path, target_object = input.split(",")
        image = self.processor(Image.open(image_path).convert("RGB"))
        image = np.asarray(image)
        images = torch.from_numpy(image.copy()).permute(2,0,1).to(self.device)
        texts_input = [[target_object.strip() if target_object.endswith('.') else (target_object + '.')]]
        batch_inputs = [{'image': images, 'groundings': {'texts':texts_input}, 'height': images.shape[1], 'width': images.shape[2]}]
        # grounding first
        outputs = xdmodel.model.model.evaluate_grounding(batch_inputs, None)
        grd_mask_ = outputs[-1]['grounding_mask']
        color = [252/255, 91/255, 129/255]
        visual = Visualizer(image, metadata=metedata)
        demo = visual.draw_binary_mask((grd_mask_>0).float().cpu().numpy()[0], color=color, text=target_object)
        res = demo.get_image()

        updated_image_path = get_new_image_name(image_path, func_name="referseg")
        real_image = Image.fromarray(res)  # get default the index0 image
        real_image.save(updated_image_path)
        return updated_image_path

class BLIPVQA:
    """ 
    NOTE: We will REPLACE this vqa model with our X-Decoder soon!!!
    """
    def __init__(self, device):
        print("Initializing BLIP VQA to %s" % device)
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)

    def get_answer_from_question_and_image(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert('RGB')
        print(F'BLIPVQA :question :{question}')
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer

class ConversationBot:
    def __init__(self):
        print("Initializing X-Chat")
        self.llm = OpenAI(engine="text003", temperature=0)
        self.edit = ImageEditing(device="cuda:0")
        self.i2t = ImageCaptioning(device="cuda:0")
        self.iret = ImageRetrieval(device="cuda:0")
        self.refi2t = ReferringImageCaptioning(device="cuda:0")
        self.refseg = ReferringImageSegmentation(device="cuda:0")
        self.t2i = T2I(device="cuda:0")
        self.BLIPVQA = BLIPVQA(device="cuda:0")
        self.pix2pix = Pix2Pix(device="cuda:0")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = [
            Tool(name="Get Photo Description", func=self.i2t.inference,
                 description="useful when you want to know what is inside the photo. receives image_path as input. "
                             "The input to this tool should be a string, representing the image_path. "),  # can be replaced by x-decoder
            Tool(name="Get Referring Photo Description", func=self.refi2t.inference,
                 description="useful when you want to know what is about a specific object or region in the photo. like: describe the bird in the image or describe the mountain in the image. receives image_path as input. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the object need to be described. "),  # can be replaced by x-decoder                             
            Tool(name="Get Referring Image Segementation", func=self.refseg.inference,
                 description="useful when you want to segment a specific object or region in the photo. like: segment the bird in the image or the mountain in the image. receives image_path as input. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the object need to be segmented. "),  # can be replaced by x-decoder                                                          
            Tool(name="Retrieve Image From A Pool Given User Input Text", func=self.iret.inference,
                 description="useful when you want to find or retrieve an image from the image pool given an user input text. like: find me an image of an object or something, or give me an image that includes some objects. "
                             "The input to this tool should be a string, representing the text used to retrieve image. "), 
            Tool(name="Generate Image From User Input Text", func=self.t2i.inference,
                 description="useful when you want to generate an image from a user input text and save it to a file. like: generate an image of an object or something, or generate an image that includes some objects. "
                             "The input to this tool should be a string, representing the text used to generate image. "), 
            Tool(name="Remove Something From The Photo", func=self.edit.remove_part_of_image,
                 description="useful when you want to remove and object or something from the photo from its description or location. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the object need to be removed. "),  # can be replaced by instruct-x-decoder
            Tool(name="Replace Something From The Photo", func=self.edit.replace_part_of_image,
                 description="useful when you want to replace an object from the object description or location with another object from its description. "
                             "The input to this tool should be a comma seperated string of three, representing the image_path, the object to be replaced, the object to be replaced with "), # can be replaced by instruct-x-decoder
            Tool(name="Change Something In The Photo", func=self.pix2pix.inference,
                 description="useful when you want to change the style of the image or object in the image to be like the text. like: make it look like a painting. or make it like a robot. or change the curtain to red. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the text. "),  # can be replaced by instruct-pix2pix + x-decoder
            Tool(name="Answer Question About The Image", func=self.BLIPVQA.get_answer_from_question_and_image,
                 description="useful when you need an answer for a question based on an image. like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
                             "The input to this tool should be a comma seperated string of two, representing the image_path and the question"), # can be replaced by x-decoder
            ]
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': X_CHAT_PREFIX, 'format_instructions': X_CHAT_FORMAT_INSTRUCTIONS, 'suffix': X_CHAT_SUFFIX}, )

    def run_text(self, text, state):
        print("===============Running run_text =============")
        print("Inputs:", text, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text})
        print("======>Current memory:\n %s" % self.agent.memory)
        response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print("Outputs:", state)
        return state, state

    def run_image(self, image, state, txt):
        print("===============Running run_image =============")
        print("Inputs:", image, state)
        print("======>Previous memory:\n %s" % self.agent.memory)
        image_filename = os.path.join('image', str(uuid.uuid4())[0:8] + ".png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.i2t.inference(image_filename)
        Human_prompt = "\nHuman: provide a figure named {}. The description is: {}. This information helps you to understand this image, but you should use tools to finish following tasks, " \
                       "rather than directly imagine from my description. If you understand, say \"Received\". \n".format(image_filename, description)
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        print("======>Current memory:\n %s" % self.agent.memory)
        state = state + [(f"![](/file={image_filename})*{image_filename}*", AI_prompt)]
        print("Outputs:", state)
        return state, state, txt + ' ' + image_filename + ' '

if __name__ == '__main__':
    gr.close_all()    
    bot = ConversationBot()
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="X-Chat")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear️")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

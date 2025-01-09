"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

import sys
import os
import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig

# How to use:  
# python tools/inference/torch_inf.py -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml 
# -r /data1/zy/code/d_fine/output/dfine_hgnetv2_s_custom_ft_copper/best_stg2.pth 
# --input /data1/zy/dataset/cooper001/newsplit/coco/test2017/ --device cuda:0 --output test_vis
# remember to set --output to specify the output location


category_colors = [
    (0, 0, 0),        # 灰色
    (145, 209, 79),  # 绿色
    (0, 176, 240),  # 蓝色
    (255, 255, 0),     # 黄色
    (255, 255, 255)         # 青色
]

label_category = ['Inlet', 'Slightshort', 'Generalshort', 'Severeshort', 'Outlet']

def draw(images, labels, boxes, scores, file_path, output_path, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):

            # draw.rectangle(list(b), outline='red')
            color = category_colors[lab[j].item()]
            draw.rectangle(list(b), outline=color, width=2)

            # 计算文本的宽度和高度
            text = f"{label_category[lab[j].item()]}{round(scrs[j].item()*100, 1)}"

            # 计算文本的边界框
            (label_width, label_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

            # 绘制白色背景矩形
            draw.rectangle([b[0], b[1], b[0] + label_width-2, b[1] + label_height], fill='white')

            draw.text(
                (b[0]+2, b[1]-2),
                text=f"{label_category[lab[j].item()]} {round(scrs[j].item()*100, 1)}",
                fill=(84, 74, 98),
            )

        # im.save('torch_results.jpg')
        # 找到最后一个斜杠的位置
        last_slash_index = file_path.rfind('/')

        # 提取文件名
        if last_slash_index != -1:
            file_name = file_path[last_slash_index + 1:]  # 从最后一个斜杠之后开始提取
        else:
            file_name = file_path  # 如果没有斜杠，整个字符串就是文件名

        # print(file_name)  # 输出: file.txt
        output = os.path.join(output_path, file_name)
        im.save(output)


def process_image(model, device, file_path, output_path):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, file_path, output_path)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")

# Function to process files
def process_file(model, device, file_path, output_path):
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path, output_path)
        print(f"Image processing complete for {file_path}.")
    elif os.path.splitext(file_path)[-1].lower() in ['.mp4', '.avi', '.mov']:  # Assuming video formats
        # Process as video
        process_video(model, device, file_path)
        print(f"Video processing complete for {file_path}.")
    else:
        print(f"Unsupported file format for {file_path}.")
        
def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Check if the input is a directory
    if os.path.isdir(file_path):
        # Iterate over all files in the directory
        for filename in os.listdir(file_path):
            full_path = os.path.join(file_path, filename)
            process_file(model, device, full_path, output_path)
    else:
        # Process the single file
        process_file(model, device, file_path, output_path)

    # if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
    #     # Process as image
    #     process_image(model, device, file_path)
    #     print("Image processing complete.")
    # else:
    #     # Process as video
    #     process_video(model, device, file_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()
    main(args)

import cv2
import requests
import torch
import numpy as np
import torchvision.transforms as T

from io import BytesIO

from .model import Net
from PIL import Image


def _output(img_url, best_model_pth):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # device를 설정합니다. cuda가 존재한다면 cuda, 존재하지 않는다면 cpu
    transform = T.Resize(size=(30, 30)) # img를 resize를 하기 위한 transoform 지정

    # 모델을 불러오고, 학습된 모델의 가중치를 불러옵니다.
    model = Net(n_classes=43)
    model_fn = best_model_pth # model_pth를 불러온다.
    model.load_state_dict(torch.load(model_fn, device), strict=False) # 
    model.eval()

    # 링크 형식 이미지를 불러옵니다.
    img_url = requests.get(img_url)
    img = Image.open(BytesIO(img_url.content))
    img = np.array(img)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = torch.FloatTensor(img)
    img = img.permute(2, 0, 1)
    img = transform(img)
    img = img.unsqueeze(0)
    
    pred_img = model(img)
    pred_img = torch.argmax(pred_img, dim=-1)
    pred_img = pred_img.tolist()
    pred_img = max(pred_img, key=pred_img.count)
    pred_img = abs(int(pred_img))

    pred_sentence = pred_img2num(pred_img)

    return pred_sentence


# Model의 Output에 따른 결과값.
def pred_img2num(pred_img):
    if pred_img == 0:
        return "최대 20km/h 속도로 운행"
    elif pred_img == 1:
        return "최대 30km/h 속도로 운행"
    elif pred_img == 2:
        return "최대 50km/h 속도로 운행"
    elif pred_img == 3:
        return "최대 60km/h 속도로 운행"
    elif pred_img == 4:
        return "최대 70km/h 속도로 운행"
    elif pred_img == 5:
        return "최대 80km/h 속도로 운행"
    elif pred_img == 6:
        return "최대 속도 제한 규정 종료"
    elif pred_img == 7:
        return "최대 100km/h 속도로 운행"
    elif pred_img == 8:
        return "최대 120km/h 속도로 운행"
    elif pred_img == 9:
        return "차량 추월 금지"
    elif pred_img == 10:
        return "3.5톤 이상 차량 추월 금지"
    elif pred_img == 11:
        return "다음 교차로에서 우회전"
    elif pred_img == 12:
        return "현재 주행하는 도로가 우선권 지님"
    elif pred_img == 13:
        return "상대방 차량에 주행 우선권 제공"
    elif pred_img == 14:
        return "우선 완전 정지 그리고 노선 양보"
    elif pred_img == 15:
        return "모든 종류의 차량 진입 금지"
    elif pred_img == 16:
        return "3.5톤 이상 차량 진입 금지"
    elif pred_img == 17:
        return "진입 금지"
    elif pred_img == 18:
        return "전방에 방해물이나 사고등 주의를 요함"
    elif pred_img == 19:
        return "전방에 왼쪽 방향 깊은 커브 주의"
    elif pred_img == 20:
        return "전방에 오른쪽 방향 깊은 커브 주의"
    elif pred_img == 21:
        return "이중 곡선(처음에는 왼쪽)"
    elif pred_img == 22:
        return "전방에 고르지 못한 길 주의"
    elif pred_img == 23:
        return "전방에 도로 미끄러움"
    elif pred_img == 24:
        return "전방의 차선 왼쪽을 좁아짐"
    elif pred_img == 25:
        return "전방에 작업 장소 주의"
    elif pred_img == 26:
        return "전방에 교통 신호 제어 존재"
    elif pred_img == 27:
        return "전방에 보행자 주의"
    elif pred_img == 28:
        return "전방에 어린이 주의"
    elif pred_img == 29:
        return "전방에 자전거 주의"
    elif pred_img == 30:
        return "전방에 눈으로 인한 미끄러움 주의"
    elif pred_img == 31:
        return "야생동물 출현 주의"
    elif pred_img == 32:
        return "모든 관련속도 및 추월 금지 종료"
    elif pred_img == 33:
        return "우회전 가능"
    elif pred_img == 34:
        return "좌회전 가능"
    elif pred_img == 35:
        return "직진 가능"
    elif pred_img == 36:
        return "직진 및 우회전 가능"
    elif pred_img == 37:
        return "직진 및 좌회전 가능"
    elif pred_img == 38:
        return "우측으로 유턴 가능"
    elif pred_img == 39:
        return "좌측으로 유턴 가능"
    elif pred_img == 40:
        return "전방에 회전 교차로"
    elif pred_img == 41:
        return "차량 추월 금지 제한 종료"
    elif pred_img == 42:
        return "3.5톤 이상 차량 추월 금지 제한 종료"
    else:
        return "존재하지 않는 표지판 입니다"
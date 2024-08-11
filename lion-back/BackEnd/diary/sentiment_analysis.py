import sys
import requests
import json

from django.utils import timezone
from datetime import timedelta
from .models import PrivateDiary

from pathlib import Path
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np

import os

# Naver 감정분석 API 적용 함수
def sentimentAnalysis(content) :

    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    url="https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze"

    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/json"
    }

    data = {
        "content": content
    }

    response = requests.post(url, data=json.dumps(data), headers=headers) # 감정 분석 적용
    result = json.loads(response.text) # 분석 결과를 python dictionary 형태로 변환
    rescode = response.status_code

    # 감정분석 API 결과값 중 필요한 데이터만 선별해서 반환
    if(rescode == 200):

        sentiment = result["document"]["sentiment"]
        confidence = result["document"]["confidence"]

        highlights = []
        for sentence in result["sentences"] :
            
            sentence_result = {
                "content": sentence["content"],
                "sentiment": sentence["sentiment"],
                "confidence": sentence["confidence"]
            }

            highlights.append(sentence_result)

        return sentiment, confidence, highlights 
    
    else:
        print("Error occured during sentiment analysis")


# Private Diary에 저장된 하이라이트 기반, 부정적 문장을 수집하여 반환
def collect_negative_sentences(user):

    # 최근 30일에 해당하는 데이터만 수집 대상으로 함
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=30)

    # PrivateDiary 중 요청을 보낸 사용자의 최근 30일 일기만 불러옴  
    diaries = PrivateDiary.objects.filter(user=user, date__range=[start_date, end_date])
    
    # 하이라이트 분석 결과값 활용 부정적 수치가 높은 문장만 수집
    negative_sentences = []
    for diary in diaries:

        sentiment, confidence, highlights = sentimentAnalysis(diary.body)

        for sentence in highlights:

            if sentence["sentiment"] == "negative" :
                negative_sentences.append(sentence["content"])
    # print(negative_sentences)
    return negative_sentences


# BERTClassifier 클래스 정의
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # BERT의 두 번째 출력은 pooled_output입니다.
        if self.dr_rate:
            pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


# 모델 파일 경로 설정
# model_path = "/Users/hybyu/lion-hackathon-감정분석 추가/sentiment-analysis/bert_model.pth"
def Kobert_sentiment_analysis(sentences) :

    current_folder_path = Path.cwd()
    file_name = "diary/bert_model.pth"
    file_path = current_folder_path / file_name
    model_path = file_path.as_posix()
    model_path = model_path[2:]

    # KoBERT 모델과 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    bertmodel = BertModel.from_pretrained('monologg/kobert')

    # 저장된 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

    # 모델의 state_dict를 로드할 때, 불일치를 무시하고 강제로 로드하는 코드
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            new_state_dict[k] = v
        # else:
        #     print(f"Skipping key {k} because it does not exist in the model's state_dict")

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    result = []
    for sentence in sentences :

        # 예측 함수 정의
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask, token_type_ids)
            logits = output.cpu().numpy()

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy().flatten()

        # label_dict = {0: "분노", 1: "불안", 2: "슬픔"}
        # prediction = label_dict[np.argmax(probs)]

        analysis = {'sentence':sentence, 'anger':probs[0], 'anxiety':probs[1], 'sadness':probs[2]}
        result.append(analysis)

    average_anger = sum(analysis['anger'] for analysis in result) / len(result)
    average_anxiety = sum(analysis['anxiety'] for analysis in result) / len(result)
    average_sadness = sum(analysis['sadness'] for analysis in result) / len(result)
    average_result = {'average_anger':average_anger * 100, 'average_anxiety':average_anxiety * 100, 'average_sadness':average_sadness * 100}

    return average_result



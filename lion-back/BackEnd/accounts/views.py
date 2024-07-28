from rest_framework.views import APIView
from .serializers import *
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer, TokenRefreshSerializer
from rest_framework import status, viewsets
from rest_framework.response import Response
from django.contrib.auth import authenticate
from django.shortcuts import get_object_or_404
from BackEnd.settings import SECRET_KEY
from rest_framework.permissions import IsAuthenticated
from .models import User
import jwt
from django.http import HttpResponseRedirect
from rest_framework.permissions import AllowAny
from BackEnd import settings 
from django.contrib.sites.shortcuts import get_current_site


class RegisterAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            #토큰 접근
            token = TokenObtainPairSerializer.get_token(user)
            refresh_token = str(token)
            access_token = str(token.access_token)

            res = Response(
                {
                    "user": serializer.data,
                    "message": "register success",
                    "token": {
                        "access": access_token,
                        "refresh": refresh_token,
                    },
                },
                status=status.HTTP_200_OK,
            )

            # jwt 토큰을 쿠키에 저장
            res.set_cookie("access", access_token, httponly=True)
            res.set_cookie("refresh", refresh_token, httponly=True)

            return res
        else:
            print(serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    

class LogInAPIView(APIView):
    permission_classes = [AllowAny]

    # 로그인
    def post(self, request):
    	# 유저 인증
        user = authenticate(
            email=request.data.get("email"), password=request.data.get("password")
        )
        # 이미 회원가입 된 유저일 때
        if user is not None:
            serializer = UserSerializer(user)
            # jwt 토큰 접근
            token = TokenObtainPairSerializer.get_token(user)
            refresh_token = str(token)
            access_token = str(token.access_token)
            res = Response(
                {
                    "user": serializer.data,
                    "message": "login success",
                    "token": {
                        "access": access_token,
                        "refresh": refresh_token,
                    },
                },
                status=status.HTTP_200_OK,
            )
            # jwt 토큰 => 쿠키에 저장
            res.set_cookie(key="access", value=access_token, httponly=True, secure=settings.SECURE_COOKIE, samesite='Lax')
            res.set_cookie(key="refresh", value=refresh_token, httponly=True, secure=settings.SECURE_COOKIE, samesite='Lax')
            return res
        else:
            return Response(status=status.HTTP_400_BAD_REQUEST) #존재하지 않는 유저잆니다. 회원가입하세요!!

    
class LogOutView(APIView):
    permission_classes = [IsAuthenticated]

    # 로그아웃
    def post(self, request):
        # 쿠키에 저장된 토큰 삭제 => 로그아웃 처리
        response = Response({
            "message": "Logout success"
            }, status=status.HTTP_202_ACCEPTED)
        response.delete_cookie("access")
        response.delete_cookie("refresh")
        return response
    
# jwt 토근 인증 확인용 뷰셋
# Header - Authorization : Bearer <발급받은토큰>
class UserViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    queryset = User.objects.all()
    serializer_class = UserSerializer

class VerifyTokenView(APIView):
    permission_classes = [IsAuthenticated]
    # 유저 토큰 검증
    def get(self, request):
        # access 토큰 유효할 경우
        try :
            access = request.COOKIES['access']
            payload = jwt.decode(access, SECRET_KEY, algorithms=['HS256'])
            pk = payload.get('user_id')
            user = get_object_or_404(User, pk=pk)
            serializer = UserSerializer(instance=user)

            return Response(serializer.data, status=status.HTTP_200_OK)

        # access token만 만료
        except(jwt.exceptions.ExpiredSignatureError):
            return Response(status=status.HTTP_401_UNAUTHORIZED)
        
        # # refresh token도 만료
        # except(jwt.exceptions.InvalidTokenError):
        #     return Response(status=status.HTTP_401_UNAUTHORIZED)




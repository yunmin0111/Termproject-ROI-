import numpy as np
import cv2 

video_path='TheBoyz_hypeboy.mp4' 
cap=cv2.VideoCapture(video_path) #비디오 불러오기

output_size=(187,333)
fit_to='height'

fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

if not cap.isOpened():
  exit() #window가 열리지 않는다면 중지

OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.TrackerKCF_create,
  "mil": cv2.TrackerMIL_create,

}# 현재 버전에서 boosting, tld, medianflow, mosse는 지원하지 않음

tracker = OPENCV_OBJECT_TRACKERS['csrt']() #objecting tracker crst 사용



ret, img = cap.read() #첫 프레임을 읽어 img에 저장

cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True) #ROI 설정
cv2.destroyWindow('Select Window') #ROI 지정 후 intialize

# initialize tracker
tracker.init(img, rect) #첫번째 tracker를 넣어 rect로 설정한 부분을 tracking 함

top_bottom_list, left_right_list = [], []


while True:
  
  
  ret, img = cap.read() #비디오를 읽어서 img에 저장

  if not ret:
    exit()  #다 읽으면 프로그램 종료

  success, box = tracker.update(img) #rect로 설정한 이미지와 비슷한 이미지를 찾아나감

  #성공할 시 left, top, right, bottom 정의
  left, top, w, h = [int(v) for v in box]
  right = left + w
  bottom = top + h

  # left, top, right, bottom 저장
  top_bottom_list.append(np.array([top, bottom]))
  left_right_list.append(np.array([left, right]))

  # 저장한 top, bottom 갯수가 10개가 넘으면 첫번째 원소를 삭제
  if len(top_bottom_list) > 10:
    del top_bottom_list[0]
    del left_right_list[0]

  avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int) #bottom에 대한 평균 구하기
  avg_width_range = np.mean(left_right_list, axis=0).astype(np.int) #right에 대한 평균 구하기
  avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) #avg_height_range의 평균과 avg_width_range의 평균으로 리스트 생성(중심)

  scale = 1.3
  avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
  avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

  avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2]) #높이 설정
  avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2]) #너비 설정

  # fit to output aspect ratio
  if fit_to == 'width': 
    avg_height_range = np.array([
      avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
      avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
    ]).astype(np.int).clip(0, 9999)

    avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
  elif fit_to == 'height':
    avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)

    avg_width_range = np.array([
      avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
      avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
    ]).astype(np.int).clip(0, 9999)

  # crop image
  result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

  result_img = cv2.resize(result_img, output_size) #이미지 사이즈를 output 사이즈에 맞춤

  pt1 = (int(left), int(top))
  pt2 = (int(right), int(bottom))
  cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

  cv2.imshow('img', img)
  cv2.imshow('result', result_img)
  
  #비디오 저장
  out.write(result_img)
  if cv2.waitKey(1) == ord('s'): #s를 누르면 while 문을 빠져나감
    break


cap.release()
out.release()
cv2.destroyAllWindows()



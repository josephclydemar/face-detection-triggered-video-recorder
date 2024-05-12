import os
import time
import cv2
import requests
from requests_toolbelt import MultipartEncoder
import threading


REMOTE_SERVER_HOST = 'http://192.168.1.2:8500'
HTTP_REST_ENDPOINTS = {
    'authorized_users_v1': f'{REMOTE_SERVER_HOST}/api/v1/authorized_users',
    'day_records_v1': f'{REMOTE_SERVER_HOST}/api/v1/day_records',
    'detections_v1': f'{REMOTE_SERVER_HOST}/api/v1/detections',
    'authorized_users_v2': f'{REMOTE_SERVER_HOST}/api/v2/authorized_users',
    'day_records_v2': f'{REMOTE_SERVER_HOST}/api/v2/day_records',
    'detections_v2': f'{REMOTE_SERVER_HOST}/api/v2/detections',
}



def main():
    haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    capture = cv2.VideoCapture(0)
    video_writer = None
    is_recording = False
    while True:
        is_true, frame = capture.read()
        # rescaled_frame = rescale_frame(frame, scale=0.5)
        # gray_frame = cv2.cvtColor(rescaled_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=12)
        if len(faces_rect) > 0:
            if not is_recording:
                video_writer = cv2.VideoWriter(os.path.join('videos', 'new', 'new_video.avi'), cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
                is_recording = True

            for x, y, w, h in faces_rect:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), thickness=1)
            if video_writer is not None:
                video_writer.write(frame)
        else:
            if video_writer is not None:
                video_writer.release()
                is_recording = False
                if os.path.exists(os.path.join('videos', 'new', 'new_video.avi')):
                    os.rename(os.path.join('videos', 'new', 'new_video.avi'), os.path.join('videos', 'ready', f'{time.time_ns()}.avi'))

        cv2.imshow('DeepFace Practice', frame)
        if cv2.waitKey(20) == ord('d'):
            break
    if video_writer is not None:
        video_writer.release()
    capture.release()
    cv2.destroyAllWindows()



def send_recorded_videos():
    while True:
        if len(os.listdir(os.path.join('videos', 'ready'))) > 0:
            current_day_response = requests.get(HTTP_REST_ENDPOINTS['day_records_v2'])
            current_day_json = current_day_response.json()
            print('Current Day JSON: ', current_day_json)
            for filename in os.listdir(os.path.join('videos', 'ready')):
                file = open(os.path.join('videos', 'ready', filename), 'rb')
                payload = MultipartEncoder(fields={
                    'recorded_video': (filename, file, 'video/mp4'),
                    'day_record_id': current_day_json['_id'],
                })
                response = requests.post(HTTP_REST_ENDPOINTS['detections_v1'], data=payload, headers={'Content-Type': payload.content_type})
                file.close()
                print('Detection: ', response.json())
                os.remove(os.path.join('videos', 'ready', filename))

try:
    if __name__ == '__main__':
        threads = (threading.Thread(target=send_recorded_videos),)
        for t in threads:
            t.start()
        main()
except KeyboardInterrupt:
    quit()





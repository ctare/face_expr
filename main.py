import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_center(gray_img):
    moments = cv2.moments(gray_img, False)
    try:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except:
        return None


def is_close(y0, y1):
    if abs(y0 - y1) < 10:
        return True
    return False


def eye_point(img, parts, left=True):
    if left:
        eyes = [
                parts[36],
                min(parts[37], parts[38], key=lambda x: x.y),
                max(parts[40], parts[41], key=lambda x: x.y),
                parts[39],
                ]
    else:
        eyes = [
                parts[42],
                min(parts[43], parts[44], key=lambda x: x.y),
                max(parts[46], parts[47], key=lambda x: x.y),
                parts[45],
                ]
    org_x = eyes[0].x
    org_y = eyes[1].y
    if is_close(org_y, eyes[2].y):
        return None

    eye = img[org_y:eyes[2].y, org_x:eyes[-1].x]
    _, eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY), 30, 255, cv2.THRESH_BINARY_INV)

    center = get_center(eye)
    if center:
        return center[0] + org_x, center[1] + org_y
    return center


def p(img, parts, eye):
    if eye[0]:
        cv2.circle(img, eye[0], 3, (255, 255, 0), -1)
    if eye[1]:
        cv2.circle(img, eye[1], 3, (255, 255, 0), -1)

    for i in parts:
        cv2.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    cv2.imshow("me", img)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        parts = predictor(frame, dets[0]).parts()

        left_eye = eye_point(frame, parts)
        right_eye = eye_point(frame, parts, False)

        p(frame * 0, parts, (left_eye, right_eye))

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

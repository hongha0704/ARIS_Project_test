import cv2
import numpy as np

class TemplateMatching:
    def __init__(self):
        self.frame = None
        self.frame_gray = None
        self.template = None
        self.w = None
        self.h = None
        self.res = None
        self.cap = cv2.VideoCapture(2)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.src_points = np.float32([[100, 480], [510, 480],[180, 0],[423, 0]])
        self.dst_points = np.float32([[100, 480], [510, 480], [100, 0],[510, 0]])
        self.adjust_brightness = None
        self.brightness_value = 60
        self.bright_frame = None
            
    def image_preprocessing(self):
        rows, cols, _ = self.frame.shape
        affine_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.transformed_frame = cv2.warpPerspective(self.frame, affine_matrix, (cols, rows))   
        self.bright_frame = cv2.convertScaleAbs(self.transformed_frame, alpha=1, beta=self.brightness_value)
        self.frame_gray = cv2.cvtColor(self.bright_frame, cv2.COLOR_RGB2GRAY)
        self.template = cv2.imread(template_path)
        self.template = cv2.cvtColor(self.template, cv2.COLOR_RGB2GRAY)
        self.w, self.h = self.template.shape[::-1]

    def match_template(self):
        self.res = cv2.matchTemplate(self.frame_gray, self.template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(self.res >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(self.bright_frame, pt, (pt[0] + self.w, pt[1] + self.h), (0, 0, 255), 2)
            print(pt)
        cv2.imshow('test', self.bright_frame)
        
    def run(self):
        while True:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break
            self.image_preprocessing()
            self.match_template()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    template_path = "/home/beakhongha/Downloads/template.jpeg"
    tm = TemplateMatching()
    tm.run()
/*
 * OpenCVのFacemarkLBFを使って顔のパーツを検出
 * ChatGPTと会話しながら、書いてもらったコードを手直し
 */
namespace FaceAngleLBF
{
    using OpenCvSharp;
    using OpenCvSharp.Face;
    using System.Collections.Generic;
    using System;

    class Program
    {
        static void Main(string[] args)
        {
            using (var capture = new VideoCapture(0))
            using (var faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml"))
            using (var landmarkDetector = FacemarkLBF.Create())
            {
                landmarkDetector.LoadModel("lbfmodel.yaml");

                var frame = new Mat();
                while (true)
                {
                    capture.Read(frame);
                    if (frame.Empty())
                        break;

                    var gray = new Mat();
                    Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

                    var faces = faceDetector.DetectMultiScale(gray, 1.3, 5);
                    if (faces.Length > 0)
                    {
                        var faceRect = faces[0];
                        float[] faceRectF = new float[] { faceRect.X, faceRect.Y, faceRect.Width, faceRect.Height  };
                        Mat faceRectMat = new Mat(1, 4, MatType.CV_32F, faceRectF);
                        Point2f[][] landmarks;
                        landmarkDetector.Fit(gray, faceRectMat, out landmarks);

                        List<Point> lines = new List<Point>();
                        //顔の輪郭
                        DraParts(frame, landmarks, 0, 16);
                        //左眉
                        lines.Clear();
                        DraParts(frame, landmarks, 17, 21);
                        //右眉
                        DraParts(frame, landmarks, 22, 26);
                        //鼻筋
                        DraParts(frame, landmarks, 27, 30);
                        //鼻下
                        DraParts(frame, landmarks, 31, 35);
                        //左目
                        DraParts(frame, landmarks, 36, 41, true);
                        //右目
                        DraParts(frame, landmarks, 42, 47, true);
                        //唇
                        DraParts(frame, landmarks, 48, 67, true);
                    }

                    Cv2.ImShow("Webcam", frame);
                    if (Cv2.WaitKey(1) == 'q')
                        break;
                }
            }
        }
        static void DraParts(Mat frame, Point2f[][] landmarks, int start, int end, bool bClose = false )
        {
            List<Point> lines = new List<Point>();
            for (int i = start; i <= end; i++)
            {
                Point2f p = landmarks[0][i];
                frame.Circle((int)p.X, (int)p.Y, 1, Scalar.White);
                frame.PutText(i.ToString(), new Point((int)p.X, (int)p.Y), HersheyFonts.Italic, .3, Scalar.Yellow);
                lines.Add(new Point(p.X, p.Y));
            }
            frame.Polylines(new Point[][] { lines.ToArray() }, bClose, Scalar.White);
        }
    }
}
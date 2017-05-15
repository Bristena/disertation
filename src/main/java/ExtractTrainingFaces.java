import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.highgui.Highgui.imwrite;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;


public class ExtractTrainingFaces {
    private static final String IMAGE_PREFIX = "CU_Dogs/dogImages";
    private static final String POINT_PREFIX = "CU_Dogs/dogParts/{}.txt";
    private static double FACE_BOX_SCALE = 4.0;
    private static long NUM_NEGATIVE_TRAIN_SAMPLES = 4000;
    private static long NUM_NEGATIVE_TEST_SAMPLES = 3000;
    List<String> files = new ArrayList<String>();
    private List<Part> parts = new ArrayList<Part>();

    {
        Part rightEye = new Part("RIGHT_EYE", 0);
        Part leftEye = new Part("LEFT_EYE", 1);
        Part nose = new Part("NOSE", 2);
        Part rightEarTip = new Part("RIGHT_EAR_TIP", 3);
        Part rightEarBase = new Part("RIGHT_EAR_BASE", 4);
        Part headTop = new Part("HEAD_TOP", 5);
        Part leftEarBase = new Part("LEFT_EAR_BASE", 6);
        Part leftEarTip = new Part("LEFT_EAR_TIP", 7);
        parts.add(rightEarBase);
        parts.add(rightEye);
        parts.add(leftEye);
        parts.add(nose);
        parts.add(rightEarTip);
        parts.add(headTop);
        parts.add(leftEarBase);
        parts.add(leftEarTip);
    }

    public DogUtils loadDog(String pathToDog) {
        DogUtils dogUtils = new DogUtils();
        List<INDArray> partLocations = new ArrayList<>();
        try {
            BufferedReader b = new BufferedReader(new FileReader(pathToDog));
            String readLine = "";
            while ((readLine = b.readLine()) != null) {
                String[] parts = readLine.split(" ");
                double x = Double.valueOf(parts[0]);
                double y = Double.valueOf(parts[1]);
                double[] xy = {x, y};
                INDArray nd = Nd4j.create(xy);
                partLocations.add(nd);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        Map<String, INDArray> partMap = new HashMap<>();
        for (Part part : parts) {
            partMap.put(part.getPart(), partLocations.get(part.getPosition()));
        }

        INDArray ndArray = Nd4j.hstack(partLocations);
        dogUtils.setArray(ndArray);
        dogUtils.setPartMap(partMap);
        return dogUtils;
    }

    public INDArray getCenterPoint(Map<String, INDArray> partMap) {
        INDArray centerPoint = Nd4j.zeros(2);
        centerPoint = centerPoint.add(partMap.get("LEFT_EYE"));
        centerPoint = centerPoint.add(partMap.get("RIGHT_EYE"));
        centerPoint = centerPoint.add(partMap.get("NOSE"));
        centerPoint = centerPoint.div(3);

        return centerPoint;
    }

    public INDArray getCenterPointAlt(Map<String, INDArray> partMap) {
        INDArray centerPoint = Nd4j.zeros(2);
        centerPoint = centerPoint.add(partMap.get("LEFT_EYE"));
        centerPoint = centerPoint.add(partMap.get("RIGHT_EYE"));
        centerPoint = centerPoint.div(2);

        centerPoint = centerPoint.getColumn(1).add(partMap.get("NOSE").getColumn(1));
        centerPoint = centerPoint.getColumn(1).div(1);

        return centerPoint;
    }

    public boolean pointInBox(INDArray point, Box box) {

        if (point.getColumn(0).sumNumber().intValue() < Nd4j.min(box.getBoxCorners().getColumn(0)).sumNumber().intValue()
                || point.getColumn(0).sumNumber().intValue() > Nd4j.max(box.getBoxCorners().getColumn(0)).sumNumber().intValue()) {
            return false;
        } else if (point.getColumn(1).sumNumber().intValue() < Nd4j.min(box.getBoxCorners().getColumn(1)).sumNumber().intValue()
                || point.getColumn(1).sumNumber().intValue() > Nd4j.max(box.getBoxCorners().getColumn(1)).sumNumber().intValue()) {
            return false;
        }

        return true;
    }

    public Box getFaceBox(Map<String, INDArray> parts) {
        Box box = new Box();
        INDArray center = getCenterPoint(parts);

        INDArray left_eye = parts.get("LEFT_EYE");
        INDArray right_eye = parts.get("RIGHT_EYE");

        //	eye_slope = np.array([right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]])
        INDArray eye_slope = Nd4j.hstack(right_eye.getColumn(0).sub(left_eye.getColumn(0)),
                right_eye.getColumn(1).sub(left_eye.getColumn(1)));
//        eye_slope = eye_slope / np.linalg.norm(eye_slope)
        eye_slope = eye_slope.div(Nd4j.norm1(eye_slope));
        //	eye_norm = np.array([eye_slope[1] * -1, eye_slope[0]])
        INDArray eye_norm = Nd4j.hstack(eye_slope.getColumn(1).mul(-1), eye_slope.getColumn(0));

//        inter_eye_dist = np.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
        double inter_eye_dist = Math.sqrt(Math.pow(left_eye.getColumn(0).sumNumber().doubleValue()
                - right_eye.getColumn(0).sumNumber().doubleValue(), 2) +
                (Math.pow(left_eye.getColumn(1).sumNumber().doubleValue() - right_eye.getColumn(1).sumNumber().doubleValue(), 2)));
        double dist = inter_eye_dist * FACE_BOX_SCALE / 2;

        INDArray[] box_corners = {
                center.add(eye_slope.mul(dist)).add(eye_norm.mul(dist)),
                center.add(eye_slope.mul(dist)).sub(eye_norm.mul(dist)),
                center.sub((eye_slope.mul(dist)).sub(eye_norm.mul(dist))),
                center.sub(eye_slope.mul(dist)).add(eye_norm.mul(dist)),
        };

        box.setEyeSlope(eye_slope);
        box.setInterEyeDist(inter_eye_dist);
        box.setBoxCorners(Nd4j.vstack(box_corners));

        return box;
    }

    public Box getRandomBox(Mat image, Map<String, INDArray> parts) {
        Box faceBox = getFaceBox(parts);

        while (true) {
            java.util.Random random = new java.util.Random();
            int rowsRandom = random.nextInt(image.rows());
            int colsRandom = random.nextInt(image.cols());
            INDArray center = Nd4j.create(new double[]{(double) rowsRandom, (double) colsRandom});
            INDArray slope = Nd4j.create(new double[]{random.nextInt(100), random.nextInt(100)});
            slope = slope.div(Nd4j.norm1(slope));
            INDArray norm = Nd4j.create(new double[]{slope.getColumn(1).mul(-1).sumNumber().doubleValue(),
                    slope.getColumn(0).sumNumber().doubleValue()});
            int Low = 10;
            int High = 100;
            int dist = random.nextInt(High - Low) + Low;


            if (!pointInBox(center, faceBox)) {
                Box box = new Box();
                INDArray[] box_corners = {
                        center.add(slope.mul(dist)).add(norm.mul(dist)),
                        center.add(slope.mul(dist)).sub(norm.mul(dist)),
                        center.sub((slope.mul(dist)).sub(norm.mul(dist))),
                        center.sub(slope.mul(dist)).add(norm.mul(dist)),
                };
                box.setInterEyeDist(dist);
                box.setEyeSlope(slope);
                box.setBoxCorners(Nd4j.vstack(box_corners));
                return box;
            }

        }
    }


    public List<String> getTrainingList() {
        List<String> train_images = readFromFile("path-ul catre training file");
        return train_images;
    }

    public List<String> getTestingList() {
        List<String> test_images = readFromFile("path-ul catre test file");
        return test_images;
    }


    private List<String> getFilesInDirectory(String path) {
        File folder = new File(path);
        File[] listOfFiles = folder.listFiles();
        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile()) {
                files.add(listOfFiles[i].getAbsolutePath());
            } else if (listOfFiles[i].isDirectory()) {
                getFilesInDirectory(listOfFiles[i].getPath());
            }
        }
        return files;
    }


    public List<KeyPoint> getKeypoints(Mat image, Box box) {
        List<KeyPoint> keypoints = new ArrayList<>();
        INDArray slope = box.getEyeSlope();
        double dist = box.getInterEyeDist();
        INDArray norm = Nd4j.hstack(slope.getColumn(1).mul(-1), slope.getColumn(1));
        INDArray center = Nd4j.sum(box.getBoxCorners(), 0).div(4);
        INDArray nose = center.sub((norm.mul(dist).div(2)));
        INDArray forehead = center.add(norm.mul(dist).div(3));
        INDArray left_eye = forehead.sub(slope.mul(dist).div(FACE_BOX_SCALE));
        INDArray right_eye = forehead.add(slope.mul(dist).div(FACE_BOX_SCALE));

        double nose_scale = dist;
        double eye_scale = dist / FACE_BOX_SCALE;
        double angle = (180 - Math.atan2(norm.getColumn(0).sumNumber().doubleValue(), norm.getColumn(1).sumNumber().doubleValue()) * 180 / Math.PI) % 360;

        keypoints.add(new KeyPoint(nose.getColumn(0).sumNumber().floatValue(),
                nose.getColumn(1).sumNumber().floatValue(), (float) nose_scale, (float) angle));
        keypoints.add(new KeyPoint(forehead.getColumn(0).sumNumber().floatValue(), forehead.getColumn(1).sumNumber().floatValue(),
                (float) eye_scale, (float) angle));
        keypoints.add(new KeyPoint(left_eye.getColumn(0).sumNumber().floatValue(), left_eye.getColumn(1).sumNumber().floatValue(),
                (float) eye_scale, (float) angle));
        keypoints.add(new KeyPoint(right_eye.getColumn(0).sumNumber().floatValue(), right_eye.getColumn(1).sumNumber().floatValue(),
                (float) eye_scale, (float) angle));
        return keypoints;
    }

    public List<String> readFromFile(String path) {
        List<String> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                list.add(sCurrentLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }

    public void extractFeatures(Mat image, Box box) {
        List<KeyPoint> keyPoints = getKeypoints(image, box);
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        matOfKeyPoint.fromArray(keyPoints.toArray(new KeyPoint[]{}));
        Mat grayscale = new Mat();
        Imgproc.cvtColor(image, grayscale, COLOR_BGR2GRAY);
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        Mat outputImage = new Mat(image.rows(), image.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(grayscale, matOfKeyPoint, objectDescriptors);
        Scalar newKeypointColor = new Scalar(255, 0, 0);
        Features2d.drawKeypoints(grayscale, matOfKeyPoint, outputImage, newKeypointColor, 0);
        imwrite("keypoint_test.jpg", outputImage);
    }


//    def extract_features(image, box, slope, dist):
//    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
//    sift = cv2.SIFT()
//
//    keypoints = get_keypoints(image, box, slope, dist)
//    features = sift.compute(grayscale, keypoints)
//
//    kpimg = cv2.drawKeypoints(grayscale, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
//            cv2.imwrite('keypoint_test.jpg', kpimg)
//
//            return np.reshape(features[1], features[1].shape[0] * features[1].shape[1])
}

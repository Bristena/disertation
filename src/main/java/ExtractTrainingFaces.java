import model.Box;
import model.DogParts;
import model.Part;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

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
    private static double FACE_BOX_SCALE = 4.0;

    List<String> files = new ArrayList<String>();
    private List<Part> parts = new ArrayList<>();

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

    public DogParts loadDog(String pathToDog) {
        DogParts dogParts = new DogParts();
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

        INDArray ndArray = Nd4j.vstack(partLocations);
        dogParts.setArray(ndArray);
        dogParts.setPartMap(partMap);
        return dogParts;
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

        if ((point.getColumn(0).sumNumber().floatValue() <
                Nd4j.min(box.getBoxCorners().getColumn(0)).sumNumber().floatValue())
                || (point.getColumn(0).sumNumber().floatValue() >
                Nd4j.max(box.getBoxCorners().getColumn(0)).sumNumber().floatValue())) {
            return false;
        } else if ((point.getColumn(1).sumNumber().floatValue() <
                Nd4j.min(box.getBoxCorners().getColumn(1)).sumNumber().floatValue())
                || (point.getColumn(1).sumNumber().floatValue() > Nd4j.max(box.getBoxCorners().getColumn(1)).sumNumber().floatValue())) {
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
        INDArray eye_slope = Nd4j.hstack(right_eye.getColumn(0)
                        .sub(left_eye.getColumn(0)),
                right_eye.getColumn(1).sub(left_eye.getColumn(1)));
//        eye_slope = eye_slope / np.linalg.norm(eye_slope)
        eye_slope = eye_slope.div(Nd4j.norm2(eye_slope));
        //	eye_norm = np.array([eye_slope[1] * -1, eye_slope[0]])
        INDArray eye_norm = Nd4j.hstack(eye_slope.getColumn(1).mul(-1), eye_slope.getColumn(0));

//        inter_eye_dist = np.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
        double inter_eye_dist = Math.sqrt(
                Math.pow(left_eye.getColumn(0).sub(right_eye.getColumn(0)).sumNumber().doubleValue(), 2) +
                        (Math.pow(left_eye.getColumn(1).sub(right_eye.getColumn(1)).sumNumber().doubleValue(), 2))
        );
        double dist = inter_eye_dist * FACE_BOX_SCALE / 2;

        INDArray[] box_corners = {
                center.add(eye_slope.mul(dist)).add(eye_norm.mul(dist)),
                center.add(eye_slope.mul(dist)).sub(eye_norm.mul(dist)),
                center.sub(eye_slope.mul(dist)).sub(eye_norm.mul(dist)),
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
            slope = slope.div(Nd4j.norm2(slope));
            INDArray norm = Nd4j.create(new double[]{slope.getColumn(1).mul(-1).sumNumber().doubleValue(),
                    slope.getColumn(0).sumNumber().doubleValue()});
            int Low = 64;
            int High = 128;
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


    public List<String> getFiles(String path) {
        return readFromFile(path);
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
        INDArray norm = Nd4j.hstack(slope.getColumn(1).mul(-1), slope.getColumn(0));
        INDArray center = Nd4j.sum(box.getBoxCorners(), 0).div(4);
        INDArray nose = center.sub(norm.mul(dist).div(2));
        INDArray forehead = center.add(norm.mul(dist).div(3));
        INDArray left_eye = forehead.sub(slope.mul(dist).div(FACE_BOX_SCALE));
        INDArray right_eye = forehead.add(slope.mul(dist).div(FACE_BOX_SCALE));

        double eye_scale = dist / FACE_BOX_SCALE;
        double angle = (180 - Math.atan2(norm.getColumn(0).sumNumber().doubleValue(), norm.getColumn(1).sumNumber().doubleValue()) * 180 / Math.PI) % 360;

        keypoints.add(new KeyPoint(nose.getColumn(0).sumNumber().floatValue(),
                nose.getColumn(1).sumNumber().floatValue(), (float) dist, (float) angle));
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

    public INDArray extractFeatures(Mat image, Box box) {
        List<KeyPoint> keyPoints = getKeypoints(image, box);
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        matOfKeyPoint.fromArray(keyPoints.toArray(new KeyPoint[]{}));
        Mat grayscale = new Mat();
        Imgproc.cvtColor(image, grayscale, COLOR_BGR2GRAY);
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
        descriptorExtractor.compute(grayscale, matOfKeyPoint, objectDescriptors);
        Scalar newKeypointColor = new Scalar(255, 0, 0);
        Mat outputImage = new Mat(image.rows(), image.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
        Features2d.drawKeypoints(grayscale, matOfKeyPoint, outputImage, newKeypointColor, 4);
        imwrite("keypoint_test.jpg", outputImage);
        INDArray matrice = Nd4j.create(1, 512);
        int index = 0;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 128; col++) {
                int[] indice = {0, index};
                matrice.putScalar(indice, objectDescriptors.get(row, col)[0]);
                index++;
            }
        }
        return matrice;
    }

    public List<String> getCompletePath(String path, List<String> paths) {
        List<String> completePaths = new ArrayList<>();
        for (String s : paths) {
            completePaths.add(path + s);
        }
        return completePaths;
    }
}

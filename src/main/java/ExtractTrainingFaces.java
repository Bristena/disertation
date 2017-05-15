import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;


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
        List<INDArray> partLocations = new ArrayList<INDArray>();
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

        Map<String, INDArray> partMap = new HashMap<String, INDArray>();

        for (Part part : parts) {
            partMap.put(part.getPart(), partLocations.get(part.getPosition()));
        }

        INDArray ndArray = Nd4j.vstack(partLocations);
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

        right_eye = right_eye.getColumn(0).sub(left_eye.getColumn(0));
        INDArray eye_slope = Nd4j.vstack(right_eye.getColumn(0).sub(left_eye.getColumn(0)), right_eye.getColumn(1).sub(left_eye.getColumn(1)));
//        eye_slope = eye_slope / np.linalg.norm(eye_slope)
        eye_slope = eye_slope.div(Nd4j.norm1(eye_slope));
        INDArray eye_norm = Nd4j.vstack(eye_slope.getColumn(1).mul(eye_slope.getColumn(0)));

//        inter_eye_dist = np.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)

        INDArray inter_eye_dist = sqrt(pow(left_eye.getColumn(0).sub(right_eye.getColumn(0)), 2).add(pow(left_eye.getColumn(1).sub(right_eye.getColumn(1)), 2)));
        INDArray dist = inter_eye_dist.mul(FACE_BOX_SCALE).div(2);

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
//    public Box  getRandomBox(Imaimage, Map<String, INDArray> parts) {
//       Box faceBox = getFaceBox(parts);
//
//       while (true) {
//           INDArray random = (INDArray) new Random();
//           INDArray center = Nd4j.create();
//       }
//
//        while (True):
//        center = np.array([random.randrange(image.shape[0]), random.randrange(image.shape[1])])
//        slope = np.array([random.randrange(100), random.randrange(100)])
//        slope = slope / np.linalg.norm(slope)
//        norm = np.array([slope[1] * -1, slope[0]])
//        dist = random.randrange(64, 128)
//
//        if not point_in_box (center, face_box):
//        return np.array([
//                center + (slope * dist) + (norm * dist),
//        center + (slope * dist) - (norm * dist),
//                center - (slope * dist) - (norm * dist),
//                center - (slope * dist) + (norm * dist),
//            ]),slope, dist
//    }


    public List<String> getTrainingList() {
     List<String>   train_images = readFromFile("path-ul catre training file");
     return  train_images;
    }

    public List<String> getTestingList(){
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


    public List<String> getKeypoints(Image image, Box box, INDArray slope, INDArray dist) {
        List<String> keypoints = new ArrayList<>();
        INDArray norm = Nd4j.vstack(slope.getColumn(1).mul(-1), slope.getColumn(1));
        INDArray center = Nd4j.sum(box.getBoxCorners(), axis=0) / 4

        return keypoints;
    }


    def get_keypoints(image, box, slope, dist):
    keypoints = []

    norm = np.array([slope[1] * -1, slope[0]])

    center = np.sum(box, axis=0) / 4

    nose = center - (norm * dist / 2)
    forehead = center + (norm * dist / 3)
    left_eye = forehead - (slope * dist / FACE_BOX_SCALE)
    right_eye = forehead + (slope * dist / FACE_BOX_SCALE)

    nose_scale = dist
            eye_scale = dist / FACE_BOX_SCALE

    angle = (180 - math.atan2(norm[0], norm[1]) * 180 / math.pi) % 360

            keypoints.append(cv2.KeyPoint(x=nose[0], y=nose[1], _size=nose_scale, _angle=angle))
            keypoints.append(cv2.KeyPoint(x=forehead[0], y=forehead[1], _size=eye_scale, _angle=angle))
            keypoints.append(cv2.KeyPoint(x=left_eye[0], y=left_eye[1], _size=eye_scale, _angle=angle))
            keypoints.append(cv2.KeyPoint(x=right_eye[0], y=right_eye[1], _size=eye_scale, _angle=angle))

            return keypoints


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
}

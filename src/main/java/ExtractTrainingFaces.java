import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ExtractTrainingFaces {
    private static final String IMAGE_PREFIX = "CU_Dogs/dogImages";
    private static final String POINT_PREFIX = "CU_Dogs/dogParts/{}.txt";
    private static double FACE_BOX_SCALE = 4.0;
    private static long NUM_NEGATIVE_TRAIN_SAMPLES = 4000;
    private static long NUM_NEGATIVE_TEST_SAMPLES = 3000;
    private List<Part> parts = new ArrayList<Part>();
    List<String> files = new ArrayList<String>();


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
        INDArray eye_slope = eye_slope.div(Nd4j.linspace()) / Nd4j..norm(eye_slope);
        INDArray eye_norm = Nd4j.create([eye_slope[1] * -1, eye_slope[0]])

        inter_eye_dist = Nd4j.sqrt((left_eye[0] - right_eye[0]) * * 2 + (left_eye[1] - right_eye[1]) **2)
        dist = inter_eye_dist * FACE_BOX_SCALE / 2

        box_corners = [
        center + (eye_slope * dist) + (eye_norm * dist),
                center + (eye_slope * dist) - (eye_norm * dist),
                center - (eye_slope * dist) - (eye_norm * dist),
                center - (eye_slope * dist) + (eye_norm * dist),
            ]

        return np.array(box_corners),eye_slope, inter_eye_dist
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
}

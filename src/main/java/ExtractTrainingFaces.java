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

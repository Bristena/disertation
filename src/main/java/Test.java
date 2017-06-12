import model.Data;
import org.opencv.core.Core;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Test {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //facial detection
//        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
//        Mat image = Highgui
//                .imread("D:\\School\\CU_Dogs/dogImages/002.Afghan_hound/Afghan_hound_00125.jpg");
//        DogParts dogParts = extractTrainingFaces.loadDog("D:\\School/CU_Dogs/dogParts/002.Afghan_hound/Afghan_hound_00125.txt");
//        Box box = extractTrainingFaces.getFaceBox(dogParts.getPartMap());
//        INDArray a = extractTrainingFaces.extractFeatures(image, box);
        String commonPath = "D:\\School\\CU_Dogs\\";
//        LoadData loadData = new LoadData();
        Utils utils = new Utils();
        List<String> trainingList = utils.readFromFile(commonPath + "dogImages\\", commonPath + "training.txt");
        List<String> testingList = utils.readFromFile(commonPath + "dogImages\\", commonPath + "testing.txt");
        LoadData loadData = new LoadData();
        long seed = System.nanoTime();
        Collections.shuffle(testingList, new Random(seed));
        Collections.shuffle(trainingList, new Random(seed));
        for (int i = 800; i < testingList.size(); i++) {
            trainingList.add(testingList.get(i));
        }
        Data train = loadData.loadData(trainingList);
        try {
            loadData.trainConvNetwork(train, "test.pk");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

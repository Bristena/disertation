import model.Box;
import model.DogParts;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import java.util.List;

public class Test {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //facial detection
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        Mat image = Highgui
                .imread("E:\\CU_Dogs/dogImages/002.Afghan_hound/Afghan_hound_00125.jpg");
        DogParts dogParts = extractTrainingFaces.loadDog("E:/CU_Dogs/dogParts/002.Afghan_hound/Afghan_hound_00125.txt");
        Box box = extractTrainingFaces.getFaceBox(dogParts.getPartMap());
        INDArray a = extractTrainingFaces.extractFeatures(image, box);
        String commonPath = "E:\\CU_Dogs\\";
        LoadData loadData = new LoadData();
        List<String> trainingList = extractTrainingFaces.getCompletePath("E:\\CU_Dogs\\dogImages\\",
                extractTrainingFaces.getFiles(commonPath + "training.txt"));
        List<String> testingList = extractTrainingFaces.getCompletePath("E:\\CU_Dogs\\dogParts\\",
                extractTrainingFaces.getFiles(commonPath + "testing.txt"));
        loadData.loadData(trainingList);

    }
}

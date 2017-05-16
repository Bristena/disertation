import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class Test {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        FaceDetector faceDetector = new FaceDetector();
//        faceDetector.run();


        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        Mat image = Highgui
                .imread("D:\\School\\CU_Dogs\\dogImages\\001.Affenpinscher\\Affenpinscher_00012.jpg");
        DogUtils dogUtils = extractTrainingFaces.loadDog("D:\\School\\CU_Dogs\\dogParts\\001.Affenpinscher\\Affenpinscher_00012.txt");
        Box box = extractTrainingFaces.getRandomBox(image, dogUtils.getPartMap());
        extractTrainingFaces.extractFeatures(image, box);
    }
}

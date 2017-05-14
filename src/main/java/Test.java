import org.opencv.core.Core;
import org.opencv.core.Mat;

public class Test {
    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        FaceDetector faceDetector = new FaceDetector();
//        faceDetector.run();

        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        extractTrainingFaces.loadDog("E:\\CU_Dogs\\dogParts\\002.Afghan_hound\\Afghan_hound_00081.txt");
    }
}

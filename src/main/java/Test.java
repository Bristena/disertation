import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

public class Test {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        //facial detection
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        Mat image = Highgui
                .imread("E:/CU_Dogs/dogImages/002.Afghan_hound/Afghan_hound_00125.jpg");
        DogUtils dogUtils = extractTrainingFaces.loadDog("E:/CU_Dogs/dogParts/002.Afghan_hound/Afghan_hound_00125.txt");
        Box box = extractTrainingFaces.getFaceBox(dogUtils.getPartMap());
        Mat a = extractTrainingFaces.extractFeatures(image, box);

        System.out.println(a);
    }
}

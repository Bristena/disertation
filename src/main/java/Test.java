import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class Test {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        FaceDetector faceDetector = new FaceDetector();
//        faceDetector.run();


        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        Mat image = Highgui
                .imread("E:\\CU_Dogs\\dogImages\\001.Affenpinscher\\Affenpinscher_00001.jpg");
        DogUtils dogUtils = extractTrainingFaces.loadDog("E:\\CU_Dogs\\dogParts\\001.Affenpinscher\\Affenpinscher_00001.txt");
        Box box = extractTrainingFaces.getRandomBox(image, dogUtils.getPartMap());
        extractTrainingFaces.extractFeatures(image, box);
    }
}

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

public class Test {
    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//        FaceDetector faceDetector = new FaceDetector();
//        faceDetector.run();
        INDArray x = Nd4j.rand(3,2);	//input
        INDArray y = Nd4j.rand(3,2);	//input

        INDArray indArray = sqrt(x);
        System.out.println(indArray);
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        extractTrainingFaces.loadDog("E:\\CU_Dogs\\dogParts\\002.Afghan_hound\\Afghan_hound_00081.txt");
    }
}

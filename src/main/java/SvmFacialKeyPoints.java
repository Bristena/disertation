import java.util.List;
import java.util.Random;

/**
 * Created by Brysu on 20.05.2017.
 */
public class SvmFacialKeyPoints {

    public void run() {
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        Random random = new Random(13131313L);
        double seed = random.nextDouble() * 0.5;
        String commonPath = "E:\\CU_Dogs\\";
        List<String> trainingList = extractTrainingFaces.getFiles(commonPath + "training.txt");
        List<String> testingList = extractTrainingFaces.getFiles(commonPath + "testing.txt");
//        List<String>
//        X_train = []
//        y_train = []
//        i = 0
    }
}

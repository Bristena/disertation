import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SvmFacialKeyPoints {
    private static long NUM_NEGATIVE_TRAIN_SAMPLES = 4000;
    private static long NUM_NEGATIVE_TEST_SAMPLES = 3000;

    public void run() {
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        Random random = new Random(13131313L);
        double seed = random.nextDouble() * 0.5;
        String commonPath = "E:\\CU_Dogs\\";
        List<String> trainingList = extractTrainingFaces.getFiles(commonPath + "training.txt");
        List<String> testingList = extractTrainingFaces.getFiles(commonPath + "testing.txt");
        List<INDArray> xTrain = new ArrayList<>();
        List<INDArray> yTrain = new ArrayList<>();

        int i = 0;
        System.out.println("BUILDING POSITIVE TRAINING EXAMPLES...");
        for (String dogFile : trainingList) {

            Mat image = Highgui.imread(dogFile);
            DogUtils dogUtils = extractTrainingFaces.loadDog(dogFile);
//            INDArray center = extractTrainingFaces.getCenterPoint(dogUtils.getPartMap());
            Box faceBox = extractTrainingFaces.getFaceBox(dogUtils.getPartMap());
            xTrain.add(extractTrainingFaces.extractFeatures(image, faceBox));
            yTrain.add(Nd4j.create(1));
            i += 1;
            if (i % 500 == 0) System.out.println(i);
        }
        for (long j = 0; j <= NUM_NEGATIVE_TRAIN_SAMPLES; j++) {
            Random train = new Random();
            String imgFile = trainingList.get(train.nextInt());
            Mat image = Highgui.imread(imgFile);
            DogUtils dogUtils = extractTrainingFaces.loadDog(imgFile);
            Box faceBox = extractTrainingFaces.getFaceBox(dogUtils.getPartMap());
            xTrain.add(extractTrainingFaces.extractFeatures(image, faceBox));
            yTrain.add(Nd4j.create(0));
            if (i % 500 == 0) System.out.println(i);
        }
        System.out.println("Fitting model");
//https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1780131/
        
//        X_train = np.array(X_train)
//        X_train = np.squeeze(X_train)
//        y_train = np.array(y_train)
//
//        model = svm.SVC(probability=True)
//        model.fit(X_train, y_train)
//
//        X_test = []
//        y_test = []
//        i = 0
//

    }
}

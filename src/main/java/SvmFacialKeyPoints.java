//import model.Box;
//import model.DogParts;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.api.shape.Shape;
//import org.nd4j.linalg.factory.Nd4j;
//import org.opencv.core.Mat;
//import org.opencv.highgui.Highgui;
//import weka.classifiers.functions.LibSVM;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
//public class SvmFacialKeyPoints {
//    private static long NUM_NEGATIVE_TRAIN_SAMPLES = 4000;
//    private static long NUM_NEGATIVE_TEST_SAMPLES = 3000;
//
//    public void run() {
//        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
////        Random random = new Random(13131313L);
////        double seed = random.nextDouble() * 0.5;
//        String commonPath = "E:\\CU_Dogs\\";
//        List<String> trainingList = getCompletePath("E:\\CU_Dogs\\dogImages\\",
//                extractTrainingFaces.getFiles(commonPath + "training.txt"));
//        List<String> testingList = getCompletePath("E:\\CU_Dogs\\dogParts\\",
//                extractTrainingFaces.getFiles(commonPath + "testing.txt"));
//        List<INDArray> xTrain = new ArrayList<>();
//        List<INDArray> yTrain = new ArrayList<>();
//
//        int i = 0;
//        System.out.println("BUILDING POSITIVE TRAINING EXAMPLES...");
//        for (String dogFile : trainingList) {
//
//            Mat image = Highgui.imread(dogFile);
//            dogFile = dogFile.replace("dogImages", "dogParts");
//            dogFile = dogFile.replace(".jpg", ".txt");
//            DogParts dogParts = extractTrainingFaces.loadDog(dogFile);
////            INDArray center = extractTrainingFaces.getCenterPoint(dogParts.getPartMap());
//            Box faceBox = extractTrainingFaces.getFaceBox(dogParts.getPartMap());
//            xTrain.add(extractTrainingFaces.extractFeatures(image, faceBox));
//            yTrain.add(Nd4j.create(1));
//            i += 1;
//            if (i % 500 == 0) System.out.println(i);
//        }
//        for (long j = 0; j <= NUM_NEGATIVE_TRAIN_SAMPLES; j++) {
//            Random train = new Random();
//            String imgFile = trainingList.get(train.nextInt(trainingList.size()));
//            Mat image = Highgui.imread(imgFile);
//            imgFile = imgFile.replace("dogImages", "dogParts");
//            imgFile = imgFile.replace(".jpg", ".txt");
//            DogParts dogParts = extractTrainingFaces.loadDog(imgFile);
//            Box faceBox = extractTrainingFaces.getRandomBox(image, dogParts.getPartMap());
//            xTrain.add(extractTrainingFaces.extractFeatures(image, faceBox));
//            yTrain.add(Nd4j.create(new double[]{0.0}));
//            if (i % 500 == 0) System.out.println(i);
//        }
//        System.out.println("Fitting model");
//        int[] shapeXTrain = {1, xTrain.size()};
//        int[] shapeYTrain = {1, yTrain.size()};
//        INDArray X_train = Nd4j.create(xTrain, shapeXTrain);
//        int[] squeeze = Shape.squeeze(X_train.shape());
//        X_train = X_train.reshape(squeeze);
//        INDArray Y_train = Nd4j.create(yTrain, shapeYTrain);
//        LibSVM svm = new LibSVM();
//        System.out.println(Y_train);
//        System.out.println("---------------------------");
//        System.out.println(X_train);
//
////https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1780131/
////
////        model = svm.SVC(probability=True)
////        model.fit(X_train, y_train)
////
////        X_test = []
////        y_test = []
////        i = 0
////
//
//    }
//
//    private List<String> getCompletePath(String path, List<String> paths) {
//        List<String> completePaths = new ArrayList<>();
//        for (String s : paths) {
//            completePaths.add(path + s);
//        }
//        return completePaths;
//    }
//}

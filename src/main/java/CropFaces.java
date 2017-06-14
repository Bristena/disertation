import model.Box;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opencv.highgui.Highgui.imwrite;

public class CropFaces {

    private int CROP_SIZE = 64;
    private int NUM_CHANNELS = 3;
    private int NUM_RAND_CROPS = 6;
    private int image_size = 128;
    private String dir = "D:\\School\\croppedImages\\";

    public INDArray cropBox(String dogPath, Box boundingBox) {
//        Mat img = Highgui.imread(dogPath);
        Mat img = Highgui
                .imread("D:\\School\\CU_Dogs/dogImages/002.Afghan_hound/Afghan_hound_00125.jpg");
        double xScale = img.size().width * 1.0 / image_size;
        double yScale = img.size().height * 1.0 / image_size;

        for (int i = 0; i < boundingBox.getBoxCorners().getColumn(0).sumNumber().intValue(); i++) {
            int[] index = {i, 0};
            double value = boundingBox.getBoxCorners().getColumn(0).getDouble(i, 0) * xScale;
            boundingBox.getBoxCorners().putScalar(index, value);
        }

        for (int i = 0; i < boundingBox.getBoxCorners().getColumn(1).sumNumber().intValue(); i++) {
            int[] index = {i, 1};
            double value = boundingBox.getBoxCorners().getColumn(1).getDouble(i, 0) * yScale;
            boundingBox.getBoxCorners().putScalar(index, value);
        }
        INDArray resizedImages = Nd4j.create(NUM_RAND_CROPS, CROP_SIZE, CROP_SIZE, NUM_CHANNELS);

        for (int i = 0; i < NUM_RAND_CROPS; i++) {
            double theta = Math.atan2(boundingBox.getEyeSlope().getColumn(0).sumNumber().doubleValue(), boundingBox.getEyeSlope().getColumn(1).sumNumber().doubleValue()) + Math.PI / 2;
            if (i != 0) {
                theta = Math.random() * 0.05;
            }
            double thetaDeg = theta * 180 / Math.PI * -1;
            double cosinus = Math.cos(theta);
            double primulParam[] = {cosinus, -1 * Math.sin(theta)};
            double alDoileaParam[] = {Math.sin(theta), cosinus};
            double imp[][] = {primulParam, alDoileaParam};
            INDArray rotationMat = Nd4j.create(imp);
            Mat imgRotate = rotate(img, thetaDeg);
            INDArray boxRotate = rotationMat.mmul(boundingBox.getBoxCorners().transpose());
            double randShiftX;
            double randShiftY;
            if (i == 0) {
                randShiftX = 0.0;
                randShiftY = 0.0;
            } else {
                randShiftX = Math.random() * 3.0 * xScale;
                randShiftY = Math.random() * 3.0 * yScale;
            }

            int xMin = (int) Math.max(Math.round((boxRotate.getDouble(0, 0) + boxRotate.getDouble(0, 1) / 2 + randShiftX)), 0.0);
            int xMax = (int) Math.min(Math.round((boxRotate.getDouble(0, 2) + boxRotate.getDouble(0, 3)) / 2 + randShiftX), imgRotate.size().width);
            int yMin = (int) Math.max(Math.round((boxRotate.getDouble(1, 0) + boxRotate.getDouble(1, 3)) / 2 + randShiftY), 0.0);
            int yMax = (int) Math.min(Math.round((boxRotate.getDouble(1, 1) + boxRotate.getDouble(1, 2)) / 2 + randShiftY), imgRotate.size().height);
            Mat croppedImage = new Mat();
            int a = 0;
            int b = 0;
            for (int j = yMin; j < yMax; j++) {
                for (int l = xMin; l < xMax; l++) {
                    croppedImage.put(a, b, imgRotate.get(j, l));
                    a++;
                    b++;
                }
            }
            Size size = new Size();
            size.set(new double[]{CROP_SIZE, CROP_SIZE});
            Mat imageResized = new Mat();
            Imgproc.resize(imageResized, croppedImage, size);
            for (int row = 0; row < CROP_SIZE; row++) {
                for (int cols = 0; cols < CROP_SIZE; cols++) {
                    int[] indx = {row, cols};
                    resizedImages.putScalar(indx, imageResized.get(row, cols)[0]);
                }
            }
        }
        return resizedImages;

    }


    public void writeCroopedFaces(List<String> fileList, INDArray x, String outputDir) throws Exception {
        UtilSaveLoadMultiLayerNetwork utilSaveLoadMultiLayerNetwork = new UtilSaveLoadMultiLayerNetwork();
        MultiLayerNetwork network = utilSaveLoadMultiLayerNetwork.load("");
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        INDArray yPredict = network.output(x);
        //partea de reshape
        for (int i = 0; i < fileList.size(); i++) {
            if (i % 100 == 0) {
                System.out.println("cropped - " + i);
            }
            int scale = image_size / 2;
            Map<String, INDArray> predPoints = new HashMap<>(); //aici sigur nu ii ok
//            predPoints.put("RIGHT_EYE", Nd4j.create(yPredict.getInt(new int[]{i,0}) * scale + scale));
//            predPoints.put("LEFT_EYE", Nd4j.create(yPredict[i] * scale + scale));
//            predPoints.put("NOSE", Nd4j.create(yPredict[i] * scale + scale));
            Box box = extractTrainingFaces.getFaceBox(predPoints);
            INDArray cropedImages = cropBox(fileList.get(i), box);
            if (cropedImages != null) {
                for (int j = 0; j < NUM_RAND_CROPS; j++) {
                    String cropFileName = dir + "c_" + i + fileList.get(i);
                    Mat outputImage = new Mat(cropedImages.rows(), cropedImages.columns(), Highgui.CV_LOAD_IMAGE_COLOR);
                    imwrite(cropFileName, outputImage);
                }
            }
        }
    }

    public Mat rotate(Mat src, double angle) {
        Mat dst = new Mat();

        Point pt = new Point((int) (src.cols() / 2.0), (int) (src.rows() / 2.0));
        Mat r = Imgproc.getRotationMatrix2D(pt, angle, 1.0);
        Imgproc.warpAffine(src, dst, r, new Size(src.cols(), src.rows()));
        return dst;
    }
}

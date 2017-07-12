import model.Box;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.opencv.core.Core;
import org.opencv.core.Mat;
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

    private INDArray cropBox(String dogPath, Box boundingBox) {
        Mat img = Highgui.imread(dogPath);
        double xScale = img.size().width * 1.0 / image_size;
        double yScale = img.size().height * 1.0 / image_size;

        for (int i = 0; i < boundingBox.getBoxCorners().size(0); i++) {
            int[] index = {i, 0};
            double value = boundingBox.getBoxCorners().getDouble(i, 0) * xScale;
            boundingBox.getBoxCorners().putScalar(index, value);
        }

        for (int i = 0; i < boundingBox.getBoxCorners().size(0); i++) {
            int[] index = {i, 1};
            double value = boundingBox.getBoxCorners().getDouble(i, 1) * yScale;
            boundingBox.getBoxCorners().putScalar(index, value);
        }
        INDArray resizedImages = Nd4j.create(NUM_RAND_CROPS, CROP_SIZE, CROP_SIZE, NUM_CHANNELS);

        for (int i = 0; i < NUM_RAND_CROPS; i++) {
            double theta = Math.atan2(boundingBox.getEyeSlope().getDouble(0), boundingBox.getEyeSlope().getDouble(1)) + Math.PI / 2;
            if (i != 0) {
                theta = Nd4j.rand(new int[] {1}, new NormalDistribution(theta, 0.05)).getDouble(0);
            }
            double thetaDeg = theta * 180 / Math.PI * -1;
            double cosinus = Math.cos(theta);
            double primulParam[] = {cosinus, -1 * Math.sin(theta)};
            double alDoileaParam[] = {Math.sin(theta), cosinus};
            double imp[][] = {primulParam, alDoileaParam};
            INDArray rotationMat = Nd4j.create(imp);
            Mat imgRotate = new Mat();
            rotate(img, imgRotate, thetaDeg);
            INDArray boxRotate = rotationMat.mmul(boundingBox.getBoxCorners().transpose());
            double randShiftX;
            double randShiftY;
            if (i == 0) {
                randShiftX = 0.0;
                randShiftY = 0.0;
            } else {
                randShiftX = Nd4j.rand(new int[] {1}, new NormalDistribution(0.0, 3.0)).getDouble(0) * xScale;
                randShiftY = Nd4j.rand(new int[] {1}, new NormalDistribution(0.0, 3.0)).getDouble(0) * yScale;
            }

            int xMin = (int) Math.max(Math.round((boxRotate.getDouble(0, 0) + boxRotate.getDouble(0, 1)) / 2 + randShiftX), 0.0);
            int xMax = (int) Math.min(Math.round((boxRotate.getDouble(0, 2) + boxRotate.getDouble(0, 3)) / 2 + randShiftX), imgRotate.size().width);
            int yMin = (int) Math.max(Math.round((boxRotate.getDouble(1, 0) + boxRotate.getDouble(1, 3)) / 2 + randShiftY), 0.0);
            int yMax = (int) Math.min(Math.round((boxRotate.getDouble(1, 1) + boxRotate.getDouble(1, 2)) / 2 + randShiftY), imgRotate.size().height);
            Mat croppedImage = new Mat(new Size(image_size, image_size), imgRotate.type());
            int a = 0;
            int b = 0;
            for (int j = yMin; j < yMax; j++) {
                for (int l = xMin; l < xMax; l++) {
                    croppedImage.put(a, b, imgRotate.get(j, l)[0], imgRotate.get(j, l)[1], imgRotate.get(j, l)[2]);
                    b++;
                }
                a++;
            }
            Size size = new Size();
            size.set(new double[]{CROP_SIZE, CROP_SIZE, 3});
            Mat imageResized = new Mat();
            Imgproc.resize(croppedImage, imageResized, size);
            for (int row = 0; row < imageResized.rows(); row++) {
                for (int cols = 0; cols < imageResized.cols(); cols++) {
                    for (int color = 0; color < imageResized.channels(); color++) {
                        int[] indx = {i, row, cols, color};
                        resizedImages.putScalar(indx, imageResized.get(row, cols)[color]);
                    }
                }
            }
        }
        return resizedImages;

    }


    public void writeCroopedFaces(List<String> fileList, INDArray x) throws Exception {
        UtilSaveLoadMultiLayerNetwork utilSaveLoadMultiLayerNetwork = new UtilSaveLoadMultiLayerNetwork();
        MultiLayerNetwork network = utilSaveLoadMultiLayerNetwork.load("C:\\Users\\bristena.vrancianu\\Desktop\\train.zip");
        ExtractTrainingFaces extractTrainingFaces = new ExtractTrainingFaces();
        INDArray yPredict = network.output(x);
        yPredict = yPredict.reshape(yPredict.size(0), (yPredict.size(1) / 2), 2);
        for (int i = 0; i < fileList.size(); i++) {
            if (i % 100 == 0) {
                System.out.println("cropped - " + i);
            }
            int scale = image_size / 2;
            Map<String, INDArray> predPoints = new HashMap<>();
            predPoints.put("RIGHT_EYE", yPredict.get(NDArrayIndex.point(i), NDArrayIndex.point(0), NDArrayIndex.all()).mul(scale).add(scale).transpose());
            predPoints.put("LEFT_EYE", yPredict.get(NDArrayIndex.point(i), NDArrayIndex.point(1), NDArrayIndex.all()).mul(scale).add(scale).transpose());
            predPoints.put("NOSE", yPredict.get(NDArrayIndex.point(i), NDArrayIndex.point(2), NDArrayIndex.all()).mul(scale).add(scale).transpose());
            Box box = extractTrainingFaces.getFaceBox(predPoints);
            INDArray cropedImages = cropBox(fileList.get(i), box);
            if (cropedImages != null) {
                for (int j = 0; j < NUM_RAND_CROPS; j++) {
                    System.out.println(fileList.get(i));
                    String cropFileName = dir + "c_" + i + ".png";
                    Mat outputImage = new Mat(cropedImages.size(1), cropedImages.size(2), 16);
                    for (int a = 0; a < cropedImages.size(1); a++) {
                        for (int b = 0; b < cropedImages.size(2); b++) {
                            double[] data = new double[3];
                            for (int c = 0; c < cropedImages.size(3); c++) {
                                data[c] = cropedImages.getDouble(j, a, b, c);
                            }
                            outputImage.put(a, b, data);
                        }
                    }
                    imwrite(cropFileName, outputImage);
                }
            }
        }
    }

    public void rotate(Mat src, Mat dst, double angle) {
        if (src.dataAddr() != dst.dataAddr()) {
            src.copyTo(dst);
        }

        angle = ((angle / 90) % 4) * 90;

        //0 : flip vertical; 1 flip horizontal
        int flip_horizontal_or_vertical = angle > 0 ? 1 : 0;
        int number = (int) Math.abs(angle / 90);

        for (int i = 0; i != number; ++i) {
            Core.transpose(dst, dst);
            Core.flip(dst, dst, flip_horizontal_or_vertical);
        }
    }
}

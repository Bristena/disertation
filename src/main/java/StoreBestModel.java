public class StoreBestModel {
    private int waitTime = 350;
    private String saveFile;
    private double bestLoss;
    private double modelEpoch = 0;


    public StoreBestModel(String saveFile) {
        this.waitTime = 350;
        this.saveFile = saveFile;
        this.bestLoss = 1e10;
        this.modelEpoch = 0;
    }


//    public void call(MultiLayerConfiguration currentNet, List<Map<String, Double>> lossHistory) throws Exception {//lossHistory  - diferenta dintre epoci
//        Map<String, Double> lastElem = lossHistory.get(lossHistory.size() - 1);
//        UtilSaveLoadMultiLayerNetwork utilSaveLoadMultiLayerNetwork = new UtilSaveLoadMultiLayerNetwork();
//        if (lastElem.get("valid_loss") < this.bestLoss) {
//            this.bestLoss = lastElem.get("valid_loss");
//            this.modelEpoch = lastElem.get("model_epoch");
//            try {
//                utilSaveLoadMultiLayerNetwork.save(currentNet, this.saveFile);
//            } catch (Exception e) {
//                System.out.println(e);
//            }
//        } else if (this.modelEpoch + this.waitTime < lastElem.get("epoch")) {
//            System.out.println("MODEL HAS NOT IMPROVED SINCE EPOCH" + this.modelEpoch + "WITH LOSS " + this.bestLoss);
//            try {
//                utilSaveLoadMultiLayerNetwork.load(currentNet, this.saveFile);
//            } catch (Exception e) {
//                throw e;
//            }
//        }
//        if (lastElem.get("epoch") == 350) {
//            utilSaveLoadMultiLayerNetwork.load(currentNet,this.saveFile);
//        }

//    }
}

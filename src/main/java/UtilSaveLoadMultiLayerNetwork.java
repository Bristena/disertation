import org.apache.log4j.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;

public class UtilSaveLoadMultiLayerNetwork {
    final static Logger logger = Logger.getLogger(UtilSaveLoadMultiLayerNetwork.class);


    public void save(MultiLayerNetwork net, String filename) throws Exception {
//        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        //Save the model
        File locationToSave = new File(filename);      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);
//        logger.warn("saved params: " + net.params());
//        logger.warn("saved params");
    }

    public MultiLayerNetwork load(String filename) throws Exception {
        //Load the model
        File locationToSave = new File(filename);      //Where to save the network. Note: the file is in .zip format - can be opened externally
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

//        logger.warn("loaded params:      " + restored.params());
        return restored;
    }

}
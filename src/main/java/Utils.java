import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Utils {

    public List<String> readFromFile(String commonPath, String file) {
        System.out.println("Reading from file: " + file);
        List<String> files = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String sCurrentLine;
            while ((sCurrentLine = br.readLine()) != null) {
                files.add(commonPath + sCurrentLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return files;
    }
}

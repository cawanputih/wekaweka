package wekafeed;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import java.io.*;


public class loaddata {
   
    public static String fileName;
    public static Instances train_data;
    public static int banyakatribut;
    public static int banyakkelas;
    
    public loaddata(String _filename) {  
        fileName=_filename;
        
        try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				
				// membaca file dengan format ARFF
				ArffReader arff = new ArffReader(reader);
                                
				// menyimpan isi file (bagian training data) dalam variabel trainData
				train_data = arff.getData();
                                
                                train_data.setClassIndex(train_data.numAttributes()-1);
                                banyakatribut=train_data.numAttributes()-1;
                                banyakkelas=train_data.numClasses();
				reader.close();
		
		} catch (IOException e) {
			e.printStackTrace();
				// kasus jika pembacaan file gagal
				System.out.println("[FAIL] Gagal mengakses datatrain: " + fileName + "\n");
				System.out.println("--------------------------------------------------------------------" + "\n");
		}
    }
    
}

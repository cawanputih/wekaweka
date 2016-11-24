
package wekafeed;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.Standardize;


public class Standardization {
    
    private Instances train_data;
    
    public Standardization(Instances train_data) {
        
        this.train_data = train_data;
        
    }
    
    
    public Instances standardize() {
        
        try {
            Standardize m_Filter = new Standardize();
            m_Filter.setInputFormat(train_data);

            if (m_Filter != null) {
                
                // normalize the converted training dataset
                m_Filter.setInputFormat(train_data);
                train_data = Filter.useFilter(train_data, m_Filter);
            
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        return train_data;
    }
}


package wekafeed;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.Normalize;


public class Normalization {
    
    private Instances train_data;
    
    public Normalization(Instances train_data) {
        
        this.train_data = train_data;
        
    }
    
    
    public Instances normalize() {
        
        try {
            Normalize m_Filter = new Normalize();
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
    
    public Instances normalizestring() {
        
  
		
        try {
	
            Normalize m_Filter = new Normalize();
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


package tucil; 

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.Filter;

import java.util.Enumeration;
import weka.core.Utils;

public class NaiveBayes050 extends AbstractClassifier{
        
    Typeatr[][] distribution;
    Typeatr classdistribution;
    boolean discretize = false;
    boolean usenormal = false;
    double[] count;
    double sumcount;
    int nclass;
    Instances Ins;
    final double defaultpres = 0.01;
    Discretize disc = null;

  
  @Override
  public void buildClassifier(Instances ins) throws Exception {
    ins.deleteWithMissingClass();
    nclass = ins.numClasses();
    Ins = new Instances(ins);

    if(discretize)
    {
        disc = new Discretize();
        disc.setInputFormat(Ins);
        Ins = Filter.useFilter(Ins, disc);
    }
    else
    {
        disc = null;
    }
      
    distribution = new Typeatr[Ins.numAttributes()-1][Ins.numClasses()];
    classdistribution = new Discreteatr(Ins.numClasses(),true);
    
    
    int index = 0;
    Enumeration enu = Ins.enumerateAttributes();
    while(enu.hasMoreElements())
    {
        Attribute att = (Attribute) enu.nextElement();
        double pres = defaultpres;
        if(att.type()== Attribute.NUMERIC)
        {
            Ins.sort(att);
            if((Ins.numInstances()>0) && !Ins.instance(0).isMissing(att))
            {
                double last = Ins.instance(0).value(att);
                double curr,delta = 0;
                int distinct = 0;
                for (int i = 1; i < Ins.numInstances(); i++) {
                    Instance currentInst = Ins.instance(i);
                    if (currentInst.isMissing(att)) {
                      break;
                    }
                    curr = currentInst.value(att);
                    if (curr != last) 
                    {
                        delta += curr - last;
                        last = curr;
                        distinct++;
                    }
                }
                if(distinct > 0)
                {
                    pres = delta/distinct;
                }
            }
        }
        
        for (int j = 0; j < Ins.numClasses(); j++) 
        {
            switch (att.type()) {
            case Attribute.NUMERIC: 
              if (!usenormal) 
              {
                distribution[index][j] = new Kernelatr(pres);
              } 
              else 
              {
                distribution[index][j] = new Normalatr(pres);
              }
              break;
            case Attribute.NOMINAL:
                distribution[index][j] = new Discreteatr(att.numValues(), true);
              break;
            default:
            throw new Exception("Attribute tidak diketahui dengan naive bayes");
            }
        }
            index++;
    }
    
    Enumeration enumins = Ins.enumerateInstances();
    while(enumins.hasMoreElements())
    {
        Instance instance = (Instance) enumins.nextElement();
        updateClassifier(instance);
    }
  }

    public void updateClassifier(Instance inst) throws Exception {
        if(!inst.classIsMissing())
        {
            Enumeration enumatt = Ins.enumerateAttributes();
            int index = 0;
            double max = inst.weight();
            while(enumatt.hasMoreElements())
            {
                Attribute att = (Attribute) enumatt.nextElement();
                if(!inst.isMissing(att))
                {
                    distribution[index][(int)inst.classValue()].addValue(inst.value(att), inst.weight());
                }
                index++;
            }
        }
        classdistribution.addValue(inst.classValue(), inst.weight());
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception 
    {
        double [] dist = distributionForInstance(instance);
        if (dist == null) 
        {
          throw new Exception("distribusi tidak terpredistribusi dengan baik");
        }
        switch (instance.classAttribute().type()) 
        {
           case Attribute.NOMINAL:
            double max = 0;
            int maxIndex = 0;

            for (int i = 0; i < dist.length; i++) 
            {
                if (dist[i] > max) 
                {
                    maxIndex = i;
                    max = dist[i];
                }
            }

            if (max > 0) 
            {
                return maxIndex;
            } 
            else 
            {
                 return Double.NaN;
            }
        case Attribute.NUMERIC:
           return dist[0];
        default:
           return Double.NaN;
       }
  }
    
  @Override
  public double [] distributionForInstance(Instance instance) 
    throws Exception { 

    double [] probs = new double[nclass];
    for (int j = 0; j < nclass; j++) {
      probs[j] = classdistribution.getProbability(j);
    }
    Enumeration enumAtts = instance.enumerateAttributes();
    int attIndex = 0;
    while (enumAtts.hasMoreElements()) 
    {
        Attribute attribute = (Attribute) enumAtts.nextElement();
        if (!instance.isMissing(attribute)) 
        {
            double temp, max = 0;
            for (int j = 0; j < nclass; j++) 
            {
                temp = Math.max(1e-75, Math.pow(distribution[attIndex][j].getProbability((int)instance.value(attribute)),Ins.attribute(attIndex).weight()));
                probs[j] *= temp;
                if (probs[j] > max) {
                  max = probs[j];
                }
                if (Double.isNaN(probs[j])) 
                {
                  throw new Exception("bukan angka woi"+ attribute.name() + ":\n"+ distribution[attIndex][j].toString());
                }
            }
            if ((max > 0) && (max < 1e-75)) 
            {
                for (int j = 0; j < nclass; j++) 
                {
                  probs[j] *= 1e75;
                }
            }
          }
        attIndex++;
    }
    Utils.normalize(probs);
    return probs;
  }
    
    
    
}
package weka.classifiers.LC;

import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

public class ALVOT
        extends AbstractClassifier
        implements OptionHandler,
        WeightedInstancesHandler, WeightedAttributesHandler, TechnicalInformationHandler{

  static final long serialVersionUID = 2487520790733881279L;

  /** The training instances used for classification. */
  protected Instances m_Train;

  /** The number of class values (or 1 if predicting numeric). */
  protected int m_NumClasses;

  /** The class attribute type. */
  protected int m_ClassType;

  protected  String[] classes;
  
  ArrayList<omegasSet> listaOSetClass = new ArrayList<omegasSet>();

  //En este array se guardan los conjuntos de rasgos complejos para cada clase
  protected ArrayList<ClassComplexTraits> model = new ArrayList<ClassComplexTraits>();

  /** the distance function used. */
  protected DistanceFunction m_SimilarityMeasure= new SimilarityMeasure();

  /*variable file path of omega parts*/
  protected String omegasFilePath="";
  /*variable n omegas parts*/
  protected int numOmegas=3;


  public String numOmegasTipText(){
    return "Indicate the number of n omega parts, this is the defualt option if you do not specify"
            + " a file path";
  }

  /**
   * Set the path
   *
   * @param omegasFilePath file path
   */
  public void setNumOmegas(int _num) {
    numOmegas=_num;
  }
  /**
   * Get the paths to omegas
   *
   * @return an array of File paths to serialized models
   * @throws IOException
   */
  public int getNumOmegas() {
    return numOmegas;
  }

  public String omegasFilePathTipText() {
    return "Specifiy a file path who contains omega parts to initilizate the algorithm";
  }

  /**
   * Set the path
   *
   * @param omegasFilePath file path
   */
  public void setOmegasFilePath(String _filePath) {
    omegasFilePath=_filePath;

  }

  /**
   * Get the paths to omegas
   *
   * @return an array of File paths to serialized models
   * @throws IOException
   */
  public String getOmegasFilePath() {
    return omegasFilePath;
  }
  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */

  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> newVector = new Vector<Option>();

    newVector.add(new Option("\tSimlarity function to use.\n"
            + "\t(default: weka.core.SimilarityMeasure.java)", "A", 1,
            "-A <classname and options>"));

    return newVector.elements();
  }

  @Override
  public void setOptions(String[] options) throws Exception {
    String LCSearchClass = Utils.getOption('A', options);
    if (LCSearchClass.length() != 0) {
      String LCSearchClassSpec[] = Utils.splitOptions(LCSearchClass);
      if (LCSearchClassSpec.length == 0) {
        throw new Exception("Invalid DistanceFunction specification string.");
      }
      String className = LCSearchClassSpec[0];
      LCSearchClassSpec[0] = "";

      setSimilarityMeasure((DistanceFunction) Utils.forName(
              DistanceFunction.class, className, LCSearchClassSpec));
    } else {
      setSimilarityMeasure(new SimilarityMeasure());
    }
  }

  @Override
  public String[] getOptions() {
    Vector<String> result;

    result = new Vector<String>();

    result.add("-A");
    result.add((m_SimilarityMeasure.getClass().getName() + " " + Utils
            .joinOptions(m_SimilarityMeasure.getOptions())).trim());

    return result.toArray(new String[result.size()]);
  }

  public String similarityMeasureTipText() {
    return "The similarity measure to use "
            + "(default: weka.core.SimilarityMeasure). ";
  }

  public DistanceFunction getSimilarityMeasure() {
    return m_SimilarityMeasure;
  }

  public void setSimilarityMeasure(DistanceFunction df) throws Exception {
    m_SimilarityMeasure = df;
  }

  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);

    // instances
    result.setMinimumNumberInstances(0);

    return result;
  }

  @Override
  public void buildClassifier(Instances instances) throws Exception {

    getCapabilities().testWithFail(instances);

    String clases = instances.classAttribute().toString();

    instances = new Instances(instances);
    instances.deleteWithMissingClass();

    m_NumClasses = instances.numClasses();
    m_ClassType = instances.classAttribute().type();
    m_Train = new Instances(instances, 0, instances.numInstances());
    m_SimilarityMeasure.setInstances(m_Train);
    double delta;
    double deltaPrime;
    double AA;


    AttributeStats[] m_AttributeStats;
    m_AttributeStats = new AttributeStats[m_Train.numAttributes()];
    m_AttributeStats[m_Train.classIndex()] = m_Train.attributeStats(m_Train.classIndex());
    AttributeStats as = m_AttributeStats[m_Train.classIndex()];
    int[] instancesPerClass = as.nominalCounts;
    classes = new String[m_Train.classAttribute().numValues()];
    for(int val=0; val<m_Train.classAttribute().numValues(); val++){
      classes[val] = m_Train.classAttribute().value(val);
    }

    //Si se tiene un ruta de archivo se utiliza, si no, se utiliza un k predefinido
    if(!omegasFilePath.equals("")) {
      //se lee archivo
      String cadena;
      FileReader file = new FileReader(omegasFilePath);
      BufferedReader b = new BufferedReader(file);
      int num_lines=0;
      while((cadena = b.readLine())!=null) {
        cadena = cadena.trim();
        if (cadena.length() == 0) {
          continue;
        } else {
          num_lines+=1;
          int numAttributes = instances.numAttributes();
          String[] div = clases.split("\\{",2);
          String[] classNames = div[1].split("\\}",2);
          /*como acordado cada linea pertenece a una clase
           * por lo que se crea las omegas partes */
          String[] div4 = cadena.split("\\:",2);
          String className = div4[0];
          testExistClass(className,classNames[0]);
          testAlreadyAdded(className,listaOSetClass);
          omegasSet set = new omegasSet();
          set.setOmegasSetFromFile(className+div4[1],numAttributes);
          /*Se agrega a la lista, la lista tendra
           * tantos elementos como lineas del archivo
           * que corresponden a el numero de clases*/
          listaOSetClass.add(set);
        }
      }
    }else {
      //busco subconjuntos de n cuando no se especifican con el archivo
      //System.out.println("Sera con n partes= "+numOmegas);

      omegasSet set = new omegasSet();
      set.setOmegasFromSize(classes[0], numOmegas, instances.classIndex());
      listaOSetClass.add(set);
      for (int i=1;i<classes.length;i++) {
        omegasSet set1 = new omegasSet();
        set1.setOmegasFromString(classes[i], set.getIndicesString(), instances.classIndex());
        listaOSetClass.add(set1);
      }
    }

  }

  private void sendException(String message) throws Exception{
    Exception m_FailReason = new WekaException(message);
    throw m_FailReason;
  }

  private void testAlreadyAdded(String _classToFind, ArrayList<omegasSet> _omegasSetList) throws Exception {
    boolean finded = false;
    int c=0;
    while (c<_omegasSetList.size() && finded==false) {
      omegasSet omega = _omegasSetList.get(c);
      if(omega.getOmegasClassName().equals(_classToFind)) {
        finded = true;
      }
      c++;
    }
    if(finded) {
      sendException("There are more than 1 class declaration in the file."
              + "\nClass Name "+_classToFind);
    }
  }
  private void testExistClass(String _className, String _allClasses) throws Exception {
    // TODO Auto-generated method stub
    String[] classes = _allClasses.split(",");
    boolean finded = false;
    int c=0;
    while (c<classes.length && finded==false) {
      if(classes[c].equals(_className)) {
        finded = true;
      }
      c++;
    }
    if(!finded) {
      sendException("There are more classes in the file than in dataset."
              + "\nClass Name: "+_className);
    }
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return predicted class probability distribution
   * @exception Exception if there is a problem generating the prediction
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
     
    double[] urna_votos = new double[classes.length];//global
    double weight;
    double votos=0;
   double conteo = 0;
      
    //para cada clase
       for(omegasSet omegasClase:listaOSetClass)
    {
         votos=0;
        String name_class =omegasClase.getOmegasClassName();
        /*Se obtienen todas las omegas partes de esa clase
       * recordemos que pueden ser n y de distinto tamaño*/
        ArrayList<omegas> sub_indices_omegas = omegasClase.getOmegasSet();
      //para cada omega hay un peso
        ArrayList<Double> omega_weights = omegasClase.getWeghts(); 
        
        
      int index = 0;
      for(int j=0; j<classes.length; j++){
        if(name_class.equals(classes[j])){;
          index = j;
          break;
        }}
       
          //Inician las comparaciones de todos los pares de instancias
            for(int i1=0; i1<m_Train.size(); i1++)
            {
                if(m_Train.get(i1).stringValue(m_Train.get(i1).classIndex()).equals(name_class))
                 {
                    for(int y=0;y<sub_indices_omegas.size();y++){
                      weight = omega_weights.get(y);
                      omegas subconjunto = sub_indices_omegas.get(y); 
                      String indices = subconjunto.getOmegas();
                      m_SimilarityMeasure.setAttributeIndices(indices);
                        
                    votos += weight*m_SimilarityMeasure.distance(instance,m_Train.get(i1)); 
                    }
                 }
            }
 
       
        urna_votos[index]=votos;
        conteo += votos;

        
    }
      
      // Al parecer para evaluar Weka pide probabilidades de todas las clases. Los votos se normalizan en el rango [0.1]
   for(int i=0; i<urna_votos.length; i++){
      urna_votos[i] = urna_votos[i] / conteo;
    }
    return urna_votos;
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    // TODO Auto-generated method stub
    return null;
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new ALVOT(), argv);
  }

  

}

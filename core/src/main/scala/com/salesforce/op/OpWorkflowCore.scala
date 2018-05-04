/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.DAG.{Layer, StagesDAG}
import com.salesforce.op.features.OPFeature
import com.salesforce.op.features.types.FeatureType
import com.salesforce.op.readers.{CustomReader, Reader, ReaderKey}
import com.salesforce.op.stages.impl.selector.ModelSelectorBase
import com.salesforce.op.stages.{FeatureGeneratorStage, OPStage, OpTransformer}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.stages.FitStagesUtil
import org.apache.spark.ml._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer
import scala.reflect.runtime.universe.WeakTypeTag

private[op] case object DAG {

  private[op] type Layer = Array[(OPStage, Int)]
  private[op] type StagesDAG = Array[Layer]

  /**
   * Computes stages DAG
   *
   * @param features array if features in workflow
   * @return unique stages layered by distance (desc order)
   */
  def compute(features: Array[OPFeature]): StagesDAG = {

    val (failures, parents) = features.map(_.parentStages()).partition(_.isFailure)

    if (failures.nonEmpty) {
      throw new IllegalArgumentException("Failed to compute stages DAG", failures.head.failed.get)
    }

    // Stages sorted by distance
    val sortedByDistance: Array[(OPStage, Int)] = parents.flatMap(_.get)

    // Stages layered by distance
    val layeredByDistance: StagesDAG = createLayers(sortedByDistance)


    // Unique stages layered by distance
    layeredByDistance
      .foldLeft(Set.empty[OPStage], Array.empty[Array[(OPStage, Int)]]) {
        case ((seen, filtered), uncleaned) =>
          // filter out any seen stages. also add distinct to filter out any duplicate stages in layer
          val unseen = uncleaned.filterNot(v => seen.contains(v._1)).distinct
          val nowSeen = seen ++ unseen.map(_._1)
          (nowSeen, filtered :+ unseen)
      }._2
  }

  /**
   * Layers Stages by distance
   *
   * @param stages stages sorted by distance
   * @return stages layered by distance
   */
  def createLayers(stages: Array[(OPStage, Int)]): StagesDAG = {
    stages.groupBy(_._2).toArray
      .map(_._2.sortBy(_._1.getOutputFeatureName))
      .sortBy(s => -s.head._2)
  }
}

/**
 * Parameters for pipelines and pipeline models
 */
private[op] trait OpWorkflowCore {

  @transient implicit protected lazy val log = LoggerFactory.getLogger(this.getClass)

  // the uid of the stage
  def uid: String

  // Model Selector
  private[op] type MS = ModelSelectorBase[_ <: Model[_], _ <: Estimator[_]]

  // the data reader for the workflow or model
  private[op] var reader: Option[Reader[_]] = None

  // final features from workflow, used to find stages of the workflow
  private[op] var resultFeatures: Array[OPFeature] = Array[OPFeature]()

  // raw features generated after data is read in and aggregated
  private[op] var rawFeatures: Array[OPFeature] = Array[OPFeature]()

  // features that have been blacklisted from use in dag
  private[op] var blacklistedFeatures: Array[OPFeature] = Array[OPFeature]()

  // stages of the workflow
  private[op] var stages: Array[OPStage] = Array[OPStage]()

  // command line parameters for the workflow stages and readers
  private[op] var parameters = new OpParams()

  private[op] def setStages(value: Array[OPStage]): this.type = {
    stages = value
    this
  }

  private[op] final def setRawFeatures(features: Array[OPFeature]): this.type = {
    rawFeatures = features
    this
  }


  /**
   * Set data reader that will be used to generate data frame for stages
   *
   * @param r reader for workflow
   * @return this workflow
   */
  final def setReader(r: Reader[_]): this.type = {
    reader = Option(r)
    checkUnmatchedFeatures()
    this
  }

  /**
   * Set input dataset which contains columns corresponding to the raw features used in the workflow
   * The type of the dataset (Dataset[T]) must match the type of the FeatureBuilders[T] used to generate
   * the raw features
   *
   * @param ds  input dataset for workflow
   * @param key key extract function
   * @return this workflow
   */
  final def setInputDataset[T: WeakTypeTag](ds: Dataset[T], key: T => String = ReaderKey.randomKey _): this.type = {
    val newReader = new CustomReader[T](key) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Right(ds)
    }
    reader = Option(newReader)
    checkUnmatchedFeatures()
    this
  }

  /**
   * Set input rdd which contains columns corresponding to the raw features used in the workflow
   * The type of the rdd (RDD[T]) must match the type of the FeatureBuilders[T] used to generate the raw features
   *
   * @param rdd input rdd for workflow
   * @param key key extract function
   * @return this workflow
   */
  final def setInputRDD[T: WeakTypeTag](rdd: RDD[T], key: T => String = ReaderKey.randomKey _): this.type = {
    val newReader = new CustomReader[T](key) {
      def readFn(params: OpParams)(implicit spark: SparkSession): Either[RDD[T], Dataset[T]] = Left(rdd)
    }
    reader = Option(newReader)
    checkUnmatchedFeatures()
    this
  }

  /**
   * Get the stages used in this workflow
   *
   * @return stages in the workflow
   */
  final def getStages(): Array[OPStage] = stages

  /**
   * Get the final features generated by the workflow
   *
   * @return result features for workflow
   */
  final def getResultFeatures(): Array[OPFeature] = resultFeatures

  /**
   * Get the list of raw features which have been blacklisted
   *
   * @return result features for workflow
   */
  final def getBlacklist(): Array[OPFeature] = blacklistedFeatures

  /**
   * Get the parameter settings passed into the workflow
   *
   * @return OpWorkflowParams set for this workflow
   */
  final def getParameters(): OpParams = parameters

  /**
   * Determine if any of the raw features do not have a matching reader
   */
  protected def checkUnmatchedFeatures(): Unit = {
    if (rawFeatures.nonEmpty && reader.nonEmpty) {
      val readerInputTypes = reader.get.subReaders.map(_.fullTypeName).toSet
      val unmatchedFeatures = rawFeatures.filterNot(f =>
        readerInputTypes
          .contains(f.originStage.asInstanceOf[FeatureGeneratorStage[_, _ <: FeatureType]].tti.tpe.toString)
      )
      require(
        unmatchedFeatures.isEmpty,
        s"No matching data readers for ${unmatchedFeatures.length} input features:" +
          s" ${unmatchedFeatures.mkString(",")}. Readers had types: ${readerInputTypes.mkString(",")}"
      )
    }
  }

  /**
   * Check that readers and features are set and that params match them
   */
  protected def checkReadersAndFeatures() = {
    require(rawFeatures.nonEmpty, "Result features must be set")
    checkUnmatchedFeatures()

    val subReaderTypes = reader.get.subReaders.map(_.typeName).toSet
    val unmatchedReaders = subReaderTypes.filterNot { t => parameters.readerParams.contains(t) }

    if (unmatchedReaders.nonEmpty) {
      log.info(
        "Readers for types: {} do not have an override path in readerParams, so the default will be used",
        unmatchedReaders.mkString(","))
    }
  }

  /**
   * Used to generate dataframe from reader and raw features list
   *
   * @return Dataframe with all the features generated + persisted
   */
  protected def generateRawData()(implicit spark: SparkSession): DataFrame

  /**
   * Fit the estimators to return a sequence of only transformers
   * Modified version of Spark 2.x Pipeline
   *
   * @param data                dataframe to fit on
   * @param stagesToFit         stages that need to be converted to transformers
   * @param persistEveryKStages persist data in transforms every k stages for performance improvement
   * @return fitted transformers
   */
  protected def fitStages(data: DataFrame, stagesToFit: Array[OPStage], persistEveryKStages: Int)
    (implicit spark: SparkSession): Array[OPStage] = {

    // TODO may want to make workflow take an optional reserve fraction
    val splitters = stagesToFit.collect{ case s: ModelSelectorBase[_, _] => s.splitter }.flatten
    val splitter = splitters.reduceOption{ (a, b) => if (a.getReserveTestFraction > b.getReserveTestFraction) a else b }
    val (train, test) = splitter.map(_.split(data)).getOrElse{ (data, spark.emptyDataFrame) }
    val hasTest = !test.isEmpty

    val dag = DAG.compute(resultFeatures)
      .map(_.filter(s => stagesToFit.contains(s._1)))
      .filter(_.nonEmpty)

    // Search for the last estimator
    val indexOfLastEstimator = dag
      .collect { case seq if seq.exists( _._1.isInstanceOf[Estimator[_]] ) => seq.head._2 }
      .lastOption

    val transformers = ListBuffer.empty[OPStage]

    dag.foldLeft((train.toDF(), test.toDF())) {
      case ((currTrain, currTest), stagesLayer) =>
        val index = stagesLayer.head._2

        val (newTrain, newTest, fitTransform) = FitStagesUtil.fitAndTransform(
          train = currTrain,
          test = currTest,
          stages = stagesLayer.map(_._1),
          transformData = indexOfLastEstimator.exists(_ < index), // only need to update for fit before last estimator
          persistEveryKStages = persistEveryKStages,
          doTest = Some(hasTest)
        )

        transformers.append(fitTransform: _*)
        newTrain -> newTest
    }
    transformers.toArray
  }


  /**
   * Returns a Dataframe containing all the columns generated up to the stop stage
   * @param stopStage last stage to apply
   * @param persistEveryKStages persist data in transforms every k stages for performance improvement
   * @return Dataframe containing columns corresponding to all of the features generated before the feature given
   */
  protected def computeDataUpTo(stopStage: Option[Int], fitted: Boolean, persistEveryKStages: Int)
    (implicit spark: SparkSession): DataFrame = {
    if (stopStage.isEmpty) {
      log.warn("Could not find origin stage for feature in workflow!! Defaulting to generate raw features.")
      generateRawData()
    } else {
      val featureStages = stages.slice(0, stopStage.get)
      log.info("Found parent stage and computing features up to that stage:\n{}",
        featureStages.map(s => s.uid + " --> " + s.getOutputFeatureName).mkString("\n")
      )
      val rawData = generateRawData()

      if (!fitted) {
        val stages = fitStages(rawData, featureStages, persistEveryKStages)
          .map(_.asInstanceOf[Transformer])
        FitStagesUtil.applySparkTransformations(rawData, stages, persistEveryKStages) // TODO use DAG transform
      } else {
        featureStages.foldLeft(rawData)((data, stage) => stage.asInstanceOf[Transformer].transform(data))
      }
    }
  }

  /**
   * Returns a dataframe containing all the columns generated up to the feature input
   *
   * @param feature             input feature to compute up to
   * @param persistEveryKStages persist data in transforms every k stages for performance improvement
   * @return Dataframe containing columns corresponding to all of the features generated before the feature given
   */
  def computeDataUpTo(feature: OPFeature, persistEveryKStages: Int = OpWorkflowModel.PersistEveryKStages)
    (implicit spark: SparkSession): DataFrame

  /**
   * Computes a dataframe containing all the columns generated up to the feature input and saves it to the
   * specified path in avro format
   */
  def computeDataUpTo(feature: OPFeature, path: String)
    (implicit spark: SparkSession): Unit = {
    val df = computeDataUpTo(feature)
    df.saveAvro(path)
  }

  /**
   * Method that cut DAG in order to perform proper CV/TS
   *
   * @param dag DAG in the workflow to be cut
   * @return (Model Selector, nonCVTS DAG -to be done outside of CV/TS, CVTS DAG -to apply in the CV/TS)
   */
  protected[op] def cutDAG(dag: StagesDAG): (Option[MS], StagesDAG, StagesDAG) = {
    if (dag.isEmpty) (None, Array.empty, Array.empty) else {
      // creates Array containing every Model Selector in the DAG
      val modelSelectorArrays = dag.flatten.collect { case (ms: MS, dist: Int) => (ms, dist) }
      val modelSelector = modelSelectorArrays.toList match {
        case Nil => None
        case List(ms) => Option(ms)
        case modelSelectors => throw new IllegalArgumentException(
          s"OpWorkflow can contain at most 1 Model Selector. Found ${modelSelectors.length} Model Selectors :" +
            s" ${modelSelectors.map(_._1).mkString(",")}")
      }

      // nonCVTS and CVTS DAGs
      val (nonCVTSDAG: StagesDAG, cVTSDAG: StagesDAG) = modelSelector.map { case (ms, dist) =>
        // Optimize the DAG by removing stages unrelated to ModelSelector
        val modelSelectorDAG = DAG.compute(Array(ms.getOutput())).dropRight(1)

        // Create the DAG without Model Selector. It will be used to compute the final nonCVTS DAG.
        val nonMSDAG: StagesDAG = {
          dag.filter(_.exists(_._2 >= dist)).toList match {
            case stages :: Nil => Array(stages.filterNot(_._1.isInstanceOf[MS]))
            case xs :+ x => xs.toArray :+ x.filterNot(_._1.isInstanceOf[MS])
          }
        }.filter(!_.isEmpty) // Remove empty layers

        // Index of first CVTS stage in ModelSelector DAG
        val firstCVTSIndex = modelSelectorDAG.toList.indexWhere(_.exists(stage => {
          val inputs = stage._1.getTransientFeatures()
          inputs.exists(_.isResponse) && inputs.exists(!_.isResponse)
        }))

        // If no CVTS stages, the whole DAG is not in the CV/TS
        if (firstCVTSIndex == -1) (nonMSDAG, Array.empty[Layer]) else {

          val cVTSDAG = modelSelectorDAG.drop(firstCVTSIndex)

          // nonCVTSDAG is the complementary DAG
          // The rule is "nonCVTSDAG = nonMSDAG - CVTSDAG"
          val nonCVTSDAG = {
            val flattenedCVTSDAG = cVTSDAG.flatten.map(_._1)
            nonMSDAG.map(_.filterNot { case (stage: OPStage, _) => flattenedCVTSDAG.contains(stage) })
              .filter(!_.isEmpty) // Remove empty layers
          }

          (nonCVTSDAG, cVTSDAG)
        }
      }.getOrElse((Array.empty[Layer], Array.empty[Layer]))
      (modelSelector.map(_._1), nonCVTSDAG, cVTSDAG)
    }
  }


  /**
   * Efficiently applies all fitted stages grouping by level in the DAG where possible
   *
   * @param rawData             data to transform
   * @param dag                 computation graph
   * @param persistEveryKStages breaks in computation to persist
   * @param spark               spark session
   * @return transformed dataframe
   */
  protected def applyTransformationsDAG(
    rawData: DataFrame, dag: StagesDAG, persistEveryKStages: Int
  )(implicit spark: SparkSession): DataFrame = {
    // A holder for the last persisted rdd
    var lastPersisted: Option[DataFrame] = None

    // Apply stages layer by layer
    dag.foldLeft(rawData) { case (df, stagesLayer) =>
      // Apply all OP stages
      val opStages = stagesLayer.collect { case (s: OpTransformer, _) => s }
      val dfTransformed: DataFrame = FitStagesUtil.applyOpTransformations(opStages, df)

      lastPersisted.foreach(_.unpersist())
      lastPersisted = Some(dfTransformed)

      // Apply all non OP stages (ex. Spark wrapping stages etc)
      val sparkStages = stagesLayer.collect {
        case (s: Transformer, _) if !s.isInstanceOf[OpTransformer] => s.asInstanceOf[Transformer]
      }
      FitStagesUtil.applySparkTransformations(dfTransformed, sparkStages, persistEveryKStages)
    }
  }


  /**
   * Looks at model parents to match parent stage for features (since features are created from the estimator not
   * the fitted transformer)
   *
   * @param feature feature want to find origin stage for
   * @return index of the parent stage
   */
  protected def findOriginStageId(feature: OPFeature): Option[Int] =
    stages.zipWithIndex.collect { case (s, i) if s.getOutput().sameOrigin(feature) => i }.headOption

}

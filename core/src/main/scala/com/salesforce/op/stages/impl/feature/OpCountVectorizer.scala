/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryModel
import com.salesforce.op.stages.sparkwrappers.specific.OpEstimatorWrapper
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.Dataset

/**
 * Wrapper around spark ml CountVectorizer for use with OP pipelines
 */
class OpCountVectorizer(uid: String = UID[OpCountVectorizer])
  extends OpEstimatorWrapper[TextList, OPVector, CountVectorizer, CountVectorizerModel](
    estimator = new CountVectorizer(),
    uid = uid
  ) {
  override val operationName: String = "countVec"

  /**
   * Set binary toggle to control the output vector values.
   * If True, all nonzero counts (after minTF filter applied) are set to 1.
   * This is useful for discrete probabilistic models that model binary events rather than integer counts.
   * Default: false
   *
   * @group setParam
   */
  def setBinary(value: Boolean): this.type = {
    getSparkMlStage().get.setBinary(value)
    this
  }

  /**
   * Set minimum number of different documents a term must appear in to be included in the vocabulary.
   * If this is an integer greater than or equal to 1, this specifies the number of documents the term must appear in;
   * if this is a double in [0,1), then this specifies the fraction of documents.
   * Default: 1.0
   *
   * @group setParam
   */
  def setMinDF(value: Double): this.type = {
    getSparkMlStage().get.setMinDF(value)
    this
  }

  /**
   * Set minimum number of times a term must appear in a document.
   * Filter to ignore rare words in a document.
   * For each document, terms with frequency/count less than the given threshold are ignored.
   * If this is an integer greater than or equal to 1, then this specifies a count
   * (of times the term must appear in the document);
   * if this is a double in [0,1), then this specifies a fraction (out of the document's token count).
   * Default: 1.0
   *
   * @group setParam
   */
  def setMinTF(value: Double): this.type = {
    getSparkMlStage().get.setMinTF(value)
    this
  }

  /**
   * Set max size of the vocabulary.
   * CountVectorizer will build a vocabulary that only considers the top vocabSize terms ordered by
   * term frequency across the corpus.
   * Default: 1 << 18
   *
   * @group setParam
   */
  def setVocabSize(value: Int): this.type = {
    getSparkMlStage().get.setVocabSize(value)
    this
  }

  override def fit(dataset: Dataset[_]): SwUnaryModel[TextList, OPVector, CountVectorizerModel] = {
    val model = super.fit(dataset)
    val vocab = model.getSparkMlStage().map(_.vocabulary).getOrElse(Array.empty[String])
    val tf = getTransientFeatures()

    val metadataCols = for {
      f <- tf
      word <- vocab
    } yield OpVectorColumnMetadata(
      parentFeatureName = Seq(f.name),
      parentFeatureType = Seq(f.typeName),
      indicatorGroup = None, // TODO do we want to test each word for label pred?
      indicatorValue = Option(word)
    )

    model.setMetadata(
      OpVectorMetadata(outputName, metadataCols, Transmogrifier.inputFeaturesToHistory(tf, stageName)).toMetadata)
    model
  }
}

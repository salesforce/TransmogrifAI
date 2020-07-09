/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.params.GeneralParams
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}

import scala.collection.mutable.ArrayBuffer

/**
 * Hack to access [[XGBoostClassifierParams]]
 */
trait OpXGBoostClassifierParams extends XGBoostClassifierParams with OpXGBoostGeneralParamsDefaults

/**
 * Hack to access [[XGBoostRegressorParams]]
 */
trait OpXGBoostRegressorParams extends XGBoostRegressorParams with OpXGBoostGeneralParamsDefaults

/**
 * XGBoost [[GeneralParams]] defaults
 */
trait OpXGBoostGeneralParamsDefaults {
  self: GeneralParams =>
  setDefault(trackerConf -> OpXGBoost.DefaultTrackerConf)
}

/**
 * Helper trait to hush XGBoost annoying logging
 */
trait OpXGBoostQuietLogging {
  Logger.getLogger("akka").setLevel(Level.WARN)
  Logger.getLogger("XGBoostSpark").setLevel(Level.WARN)
  Logger.getLogger(classOf[XGBoostClassifier]).setLevel(Level.WARN)
  Logger.getLogger(classOf[XGBoostRegressor]).setLevel(Level.WARN)
}

case object OpXGBoost {
  val DefaultTrackerConf = TrackerConf(workerConnectionTimeout = 0L, "scala")

  implicit class RichMLVectorToXGBLabeledPoint(val v: Vector) extends AnyVal {
    /**
     * Converts a [[Vector]] to a data point with a dummy label.
     *
     * This is needed for constructing a [[ml.dmlc.xgboost4j.scala.DMatrix]]
     * for prediction.
     */
    def asXGB: LabeledPoint = v match {
      case v: DenseVector => LabeledPoint(0.0f, null, v.values.map(_.toFloat))
      case v: SparseVector => LabeledPoint(0.0f, v.indices, v.values.map(_.toFloat))
    }
  }

  implicit class RichBooster(val booster: Booster) extends AnyVal {
    /**
     * Converts feature score map into a vector
     *
     * @param featureVectorSize   size of feature vectors the xgboost model is trained on
     * @return vector containing feature scores
     */
    def getFeatureScoreVector(
      featureVectorSize: Option[Int] = None, importanceType: String = "gain"
    ): Vector = {
      val featureScore = booster.getScore(featureMap = null, importanceType = importanceType)
      require(featureScore.nonEmpty, "Feature score map is empty")
      val indexScore = featureScore.map { case (fid, score) =>
        val index = fid.tail.toInt
        index -> score.toDouble
      }.toSeq
      val maxIndex = indexScore.map(_._1).max
      require(featureVectorSize.forall(_ > maxIndex), "Feature vector size must be larger than max feature index")
      val size = featureVectorSize.getOrElse(maxIndex + 1)
      Vectors.sparse(size, indexScore)
    }
  }

  /**
   * Hack to access [[ml.dmlc.xgboost4j.scala.spark.XGBoost.processMissingValues]] private method
   */
  def processMissingValues(xgbLabelPoints: Iterator[LabeledPoint], missing: Float): Iterator[LabeledPoint] =
    XGBoost.processMissingValues(xgbLabelPoints, missing)
}

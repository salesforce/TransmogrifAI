package com.salesforce.op.stages.impl.feature

import com.salesforce.op.utils.json.JsonLike
import org.apache.spark.ml.param.{IntParam, ParamValidators, Params}
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import com.twitter.algebird.Semigroup

import scala.reflect.ClassTag

/**
 * Summary statistics of a text feature
 *
 * @param valueCounts counts of feature values
 */
private[op] case class TextStats(valueCounts: Map[String, Int]) extends JsonLike

private[op] object TextStats {
  def semiGroup(maxCardinality: Int): Semigroup[TextStats] = new Semigroup[TextStats] {
    override def plus(l: TextStats, r: TextStats): TextStats = {
      if (l.valueCounts.size > maxCardinality) l
      else if (r.valueCounts.size > maxCardinality) r
      else TextStats(l.valueCounts + r.valueCounts)
    }
  }

  def empty: TextStats = TextStats(Map.empty)
}

object CategoricalDetection {
  val MaxCardinality = 100

  private[op] def partition[T: ClassTag](input: Array[T], condition: Array[Boolean]): (Array[T], Array[T]) = {
    val all = input.zip(condition)
    (all.collect { case (item, true) => item }.toSeq.toArray, all.collect { case (item, false) => item }.toSeq.toArray)
  }
}

trait MaxCardinalityParams extends Params {
  final val maxCardinality = new IntParam(
    parent = this, name = "maxCardinality",
    doc = "max number of distinct values a categorical feature can have",
    isValid = ParamValidators.inRange(lowerBound = 1, upperBound = CategoricalDetection.MaxCardinality)
  )
  final def setMaxCardinality(v: Int): this.type = set(maxCardinality, v)
  final def getMaxCardinality: Int = $(maxCardinality)
  setDefault(maxCardinality -> CategoricalDetection.MaxCardinality)
}

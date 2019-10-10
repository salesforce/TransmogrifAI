package com.salesforce.op.features

import com.salesforce.op.utils.json.JsonLike
import com.twitter.algebird.Monoid

/**
 * Summary statistics of a text feature
 *
 * @param valueCounts counts of feature values
 */
private[op] case class TextStats(valueCounts: Map[String, Int]) extends JsonLike

private[op] object TextStats {
  def monoid(maxCardinality: Int): Monoid[TextStats] = new Monoid[TextStats] {
    override def plus(l: TextStats, r: TextStats): TextStats = {
      if (l.valueCounts.size > maxCardinality) l
      else if (r.valueCounts.size > maxCardinality) r
      else TextStats(l.valueCounts + r.valueCounts)
    }

    override def zero: TextStats = TextStats.empty
  }

  def empty: TextStats = TextStats(Map.empty)
}

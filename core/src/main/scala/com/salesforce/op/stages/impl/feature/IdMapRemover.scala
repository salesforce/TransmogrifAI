package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{TextMap}
import com.salesforce.op.stages.base.unary.UnaryTransformer

class IdMapRemover(
  minUniqueTokLen: Int,
  uid: String = UID[IdMapRemover]
) extends UnaryTransformer[TextMap, TextMap](operationName = "IdMapRemover", uid = uid) {

  private var dropMap: Map[String, Boolean] = Map()

  override protected def onSetInput(): Unit = {
    super.onSetInput()
    val dist = in1.asFeatureLike.distributions
    val keys = dist.flatMap(_.key)
    val drop = dist.flatMap(_.cardEstimate).map(_.valueCounts.size < minUniqueTokLen)
    dropMap = (keys zip drop) toMap
  }

  override def transformFn: TextMap => TextMap =
    a => {
      val filteredMap = a.value.map { case (k, v) =>
        dropMap.get(k) match {
          case Some(true) => (k, "")
          case _ => (k, v)
        }
      }
      TextMap(filteredMap)
    }
}

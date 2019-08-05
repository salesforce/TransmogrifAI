package com.salesforce.op.stages.impl.feature

import enumeratum._

sealed trait LabelScaler extends EnumEntry with Serializable

object LabelScaler extends Enum[LabelScaler] {
  val values = findValues
  case object OpScalarStandardScaler extends LabelScaler
  case object MinMaxNormEstimator extends LabelScaler
  case object StandardMinEstimator extends LabelScaler
}
package com.salesforce.app

import com.salesforce.app.schema.DataClass /* << SCHEMA_IMPORT */
import com.salesforce.op.features.{FeatureBuilder => FB}
import com.salesforce.op.features.types._
import FeatureOps._

trait Features extends Serializable /* FEATURES >> */ {
  // ---------------------------------------------------------- //

  val survived = realFromInt(_.getSurvived).asResponse

  val pClass = iPickList(_.getPclass).asPredictor
  val name = text(_.getName).asPredictor
  val sex = sPickList(_.getSex).asPredictor
  val age = real(_.getAge).asPredictor
  val sibSp = int(_.getSibSp).asPredictor
  val parch = int(_.getParch).asPredictor
  val ticket = sPickList(_.getTicket).asPredictor
  val fare = real(_.getFare).asPredictor
  val cabin = sPickList(_.getCabin).asPredictor
  val embarked = sPickList(_.getEmbarked).asPredictor
}

object FeatureOps {
  def asPickList[T](f: T => Any): T => PickList = x => Option(f(x)).map(_.toString).toPickList
}

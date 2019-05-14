package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{Date, Integral}
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.date.DateTimeUtils
import org.joda.time.{DateTime => JDateTime}
import com.salesforce.op.features.types._

import scala.reflect.runtime.universe.TypeTag

/**
 * TimePeriodTransformer extracts one of a set of time periods from a date/datetime
 *
 * @param period        time period to extract from date
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @tparam I            input feature type
 */
class TimePeriodTransformer[I <: Date]
(
  val period: TimePeriod,
  uid: String = UID[TimePeriodTransformer[_]]
)(
  implicit override val tti: TypeTag[I]
) extends UnaryTransformer[I, Integral](operationName = "dateToTimePeriod", uid = uid){
  def periodFun(t: Long): Int = period match {
    case TimePeriod.DayOfMonth => new JDateTime(t, DateTimeUtils.DefaultTimeZone).dayOfMonth.get
    case TimePeriod.DayOfWeek => new JDateTime(t, DateTimeUtils.DefaultTimeZone).dayOfWeek.get
    case TimePeriod.DayOfYear => new JDateTime(t, DateTimeUtils.DefaultTimeZone).dayOfYear.get
    case TimePeriod.HourOfDay => new JDateTime(t, DateTimeUtils.DefaultTimeZone).hourOfDay.get
    case TimePeriod.MonthOfYear => new JDateTime(t, DateTimeUtils.DefaultTimeZone).monthOfYear.get
    case TimePeriod.WeekOfMonth =>
      val dt = new JDateTime(t, DateTimeUtils.DefaultTimeZone)
      dt.weekOfWeekyear.get - dt.withDayOfMonth(1).weekOfWeekyear.get
    case TimePeriod.WeekOfYear => new JDateTime(t, DateTimeUtils.DefaultTimeZone).weekOfWeekyear.get
  }

  override def transformFn: I => Integral = (i: I) => i.value.map(t => periodFun(t).toLong).toIntegral
}

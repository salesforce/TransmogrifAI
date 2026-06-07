package com.twitter.algebird

import org.json4s._

/**
 * A custom serializer for Algebird's Moments class
 *
 * Inspired from the following example: https://gist.github.com/casualjim/5130756
 * Addresses this issue in json4s: https://github.com/json4s/json4s/issues/702
 * TODO: check if the issue mentioned above is resolved
 */
class MomentsSerializer extends Serializer[Moments] {
  private val momentsClass = classOf[Moments]

  def deserialize(implicit format: Formats): PartialFunction[(TypeInfo, JValue), Moments] = {
    case (TypeInfo(`momentsClass`, _), json) =>
      json match {
      case JObject(
      JField("m0", x) ::
      JField("m1", JDouble(m1)) ::
      JField("m2", JDouble(m2)) ::
      JField("m3", JDouble(m3)) ::
      JField("m4", JDouble(m4)) :: Nil
      ) => Moments(x match {
        case JInt(m0) => m0.toLong
        case JLong(m0) => m0
        case js => throw new MappingException(s"$js can't be mapped to an Int or a Long")
      }, m1, m2, m3, m4)
    }
  }

  def serialize(implicit formats: Formats): PartialFunction[Any, JValue] = {
    case m: Moments =>
      import JsonDSL._
      ("m0" -> m.m0) ~ ("m1" -> m.m1) ~ ("m2" -> m.m2) ~ ("m3" -> m.m3) ~ ("m4" -> m.m4)
  }
}

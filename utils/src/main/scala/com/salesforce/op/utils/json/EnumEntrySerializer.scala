/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.json

import com.fasterxml.jackson.core.{JsonGenerator, JsonParser}
import com.fasterxml.jackson.databind.deser.std.StdDeserializer
import com.fasterxml.jackson.databind.ser.std.StdSerializer
import com.fasterxml.jackson.databind.{DeserializationContext, SerializerProvider}
import enumeratum.{Enum, EnumEntry}
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

import scala.reflect.ClassTag

/**
 * Serializers for [[EnumEntry]] types
 */
object EnumEntrySerializer {

  /**
   * Creates json4s serializer for Enumeratum type
   *
   * @param enum Enumeratum object
   * @tparam A Enumeratum EnumEntry type
   * @return json4s serializer for Enumeratum
   */
  def json4s[A <: EnumEntry : Manifest](enum: Enum[A]): CustomSerializer[A] = {
    new CustomSerializer[A](_ =>
      ( { case JString(s) if enum.withNameInsensitiveOption(s).isDefined => enum.withNameInsensitive(s)},
        { case x: A => JString(x.entryName) }
      )
    )
  }

  /**
   * Creates jackson serdes for Enumeratum type
   *
   * @param enum Enumeratum object
   * @tparam A Enumeratum EnumEntry type
   * @return jackson serdes for Enumeratum
   */
  def jackson[A <: EnumEntry: ClassTag](enum: Enum[A]): SerDes[A] = {
    val klazz = implicitly[ClassTag[A]].runtimeClass.asInstanceOf[Class[A]]
    val ser = new StdSerializer[A](klazz) {
      override def serialize(value: A, gen: JsonGenerator, provider: SerializerProvider): Unit = {
        gen.writeString(value.entryName)
      }
    }
    val des = new StdDeserializer[A](klazz) {
      override def deserialize(p: JsonParser, ctxt: DeserializationContext): A = {
        enum.withNameInsensitive(p.getValueAsString)
      }
    }
    new SerDes[A](klazz, ser, des)
  }

}


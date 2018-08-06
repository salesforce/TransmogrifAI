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


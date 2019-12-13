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
package com.salesforce.op.stages.impl.feature

import enumeratum.{Enum, EnumEntry}
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

import scala.util.matching.Regex

/**
 * Defines the different kinds of gender detection strategies that are possible
 *
 * We need to overwrite `toString` in order to provide serialization during the Spark map and reduce steps and then
 * the `fromString` function provides deserialization back to the `GenderDetectStrategy` class for the companion
 * transformer
 */
sealed class GenderDetectStrategy extends EnumEntry
case object GenderDetectStrategy extends Enum[GenderDetectStrategy] {
  val values: Seq[GenderDetectStrategy] = findValues
  val delimiter = " WITH VALUE "
  case class ByIndex(index: Int) extends GenderDetectStrategy {
    override def toString: String = "ByIndex" + delimiter + index.toString
  }
  case class ByLast() extends GenderDetectStrategy {
    override def toString: String = "ByLast"
  }
  case class ByRegex(pattern: Regex) extends GenderDetectStrategy {
    override def toString: String = "ByRegex" + delimiter + pattern.toString
  }
  case class FindHonorific() extends GenderDetectStrategy {
    override def toString: String = "FindHonorific"
  }

  def fromString(s: String): GenderDetectStrategy = {
    val parts = s.split(delimiter)
    val entryName: String = parts(0)
    entryName match {
      case "ByIndex" => ByIndex(parts(1).toInt)
      case "ByLast" => ByLast()
      case "ByRegex" => ByRegex(parts(1).r)
      case "FindHonorific" => FindHonorific()
      case _ => sys.error("Attempted to deserialize GenderDetectStrategy but no matching entry found.")
    }
  }

  def json4s: CustomSerializer[GenderDetectStrategy] = new CustomSerializer[GenderDetectStrategy](_ =>
    (
      { case JString(s) => GenderDetectStrategy.fromString(s) },
      { case x: GenderDetectStrategy => JString(x.toString) }
    )
  )
}

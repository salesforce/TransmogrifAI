/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.json

import enumeratum.{Enum, EnumEntry}
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

/**
 * Json4s serializer for [[EnumEntry]] types
 */
object EnumEntrySerializer {

  def apply[A <: EnumEntry : Manifest](enum: Enum[A]): CustomSerializer[A] = {
    new CustomSerializer[A](_ =>
      ( { case JString(s) if enum.withNameInsensitiveOption(s).isDefined => enum.withNameInsensitive(s)},
        { case x: A => JString(x.entryName) }
      )
    )
  }

}


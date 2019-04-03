package com.salesforce.op.stages

import com.salesforce.op.features.types.Real
import com.salesforce.op.features.types._

/**
 * @author ksuchanek
 * @since 214
 */
object Lambdas {
  def fnc0 =  (x:Real) => x.v.map(_ * 0.1234).toReal
}

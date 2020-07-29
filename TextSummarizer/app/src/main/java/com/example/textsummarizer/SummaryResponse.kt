package com.example.textsummarizer

import com.google.gson.annotations.SerializedName

class SummaryResponse {
    //@SerializedName("success")
    //var success: Boolean?=true
    @SerializedName("summary")
    var summary: String?=null
}
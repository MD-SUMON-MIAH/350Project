Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {
        let imageData = file.dataURL;
        
        var url = "http://127.0.0.1:5000/classify_image";
        $("#signRing").hide();
        $("#signMili").hide();
        $("#signCyst").hide();
        $("#RingSymptomps").hide();
        $("#CystSymptomps").hide();
        $("#MiliariaSymptomps").hide();
        

        $.post(url, {
            image_data: file.dataURL
        },function(data, status) {
            /* 
            Below is a sample response if you have two faces in an image lets say virat and roger together.
            Most of the time if there is one person in the image you will get only one element in below array
            data = [
                {
                    class: "viral_kohli",
                    class_probability: [1.05, 12.67, 22.00, 4.5, 91.56],
                    class_dictionary: {
                        lionel_messi: 0,
                        maria_sharapova: 1,
                        roger_federer: 2,
                        serena_williams: 3,
                        virat_kohli: 4
                    }
                },
                {
                    class: "roder_federer",
                    class_probability: [7.02, 23.7, 52.00, 6.1, 1.62],
                    class_dictionary: {
                        lionel_messi: 0,
                        maria_sharapova: 1,
                        roger_federer: 2,
                        serena_williams: 3,
                        virat_kohli: 4
                    }
                }
            ]
            */
            console.log(data);
            if (!data || data.length==0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();                
                $("#error").show();

                return;
            }
            let players = ["cyst", "maria_sharapova", "roger_federer", "serena_williams", "virat_kohli"];
            
           ;
            //let bestScore = -1;
           // for (let i=0;i<data.length;++i) {
           //     let maxScoreForThisClass = Math.max(...data[i].class_probability);
             //   if(maxScoreForThisClass>bestScore) {
              //      match = data[i];
              //      bestScore = maxScoreForThisClass;
              //  }
           // }
           let match=data[0]
            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();

                
                console.log(match);
                $("#resultHolder").html($(`[data-player="${match}"`).html());
                $("#resultHolder").show();
                
                if(match=="ringworm")
                {
                    $("#signRing").show();
                    $("#RingSymptomps").show();
                }
                if(match=="cyst")
                {
                    $("#signCyst").show();
                    $("#CystSymptomps").show();
                }
                
                if(match=="miliria")
                {
                    $("#signMili").show();
                    $("#MiliariaSymptomps").show();

                }
                


                $("#xSS").show();
                const aa=document.getElementsByClassName("signRing");
                aa.style.display="visible";
               // const aa2=document.getElementById("divClassTableSumon");
                //aa2.style.display="visible";
                

                

              //  let classDictionary = match.class_dictionary;
             //   for(let personName in classDictionary) {
               //     let index = classDictionary[personName];
              //      let proabilityScore = match.class_probability[index];
              //      let elementName = "#score_" + personName;
             //       $(elementName).html(proabilityScore);
             //   }
            }
            // dz.removeFile(file);            
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    console.log( "ready!" );
    $("#error").hide();
    $("#resultHolder").hide();
    //$("#divClassTable").hide();
    $("#signRing").hide();
    $("#signMili").hide();
    $("#signCyst").hide();
    //$("#divClassTableSumon").hide();
    //$("#xSS").hide();
   //$("#xS").hide();
     $("#RingSymptomps").hide();
     $("#CystSymptomps").hide();
     $("#MiliariaSymptomps").hide();
    

    init();
});

/*loading spinner*/
document.addEventListener('DOMContentLoaded', function() {
            const form = document.querySelector('form');
            const spinner = document.getElementById('spinner');
            
            if (form){   /* due to forms dynamism, check element presence before adding event listener */
                form.addEventListener('submit', function() {
                    spinner.style.display = 'block';
                });
            }
        });

/*interactive form for modeling section*/      
const classif = document.getElementById("classif");
const regr = document.getElementById("regr");
const classOpt = document.getElementById("classif-models");
const regrOpt = document.getElementById("regr-models");
if (classif){   
    classif.addEventListener("change", () => {
        if (classif.checked) {
            classOpt.style.display = "flex";
            regrOpt.style.display = "none";
        }
    });
}
if (regr){
    regr.addEventListener("change", () => {
        if (regr.checked) {
            regrOpt.style.display = "flex";
            classOpt.style.display = "none";
        }
    });
}

/*loading spinner*/
document.addEventListener('DOMContentLoaded', function() {
            const form = document.querySelector('form');
            const spinner = document.getElementById('analysis-spinner');
            
            form.addEventListener('submit', function() {
                spinner.style.display = 'block';
            });
        });

const classif = document.getElementById("classif");
const regr = document.getElementById("regr");
const classOpt = document.getElementById("classif-models");
const regrOpt = document.getElementById("regr-models");

classif.addEventListener("change", () => {
    if (classif.checked) {
        classOpt.style.display = "flex";
        regrOpt.style.display = "none";
    }
});

regr.addEventListener("change", () => {
    if (regr.checked) {
        regrOpt.style.display = "flex";
        classOpt.style.display = "none";
    }
});

/*modeling options
function toggleModels() {
    classif = document.getElementById('classif').checked
    regr = document.getElementById('regr').checked

    document.getElementById('classif-models').display = classif ? 'block' : 'none'
    document.getElementById('regr-models').display = regr ? 'block' : 'none'
}*/
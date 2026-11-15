Sys.setenv(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
dest <- normalizePath(if (length(args)) args[1] else "output/c2_c3_validation_20261113")
set.seed(20261113)
writeLines(capture.output(sessionInfo()), file.path(dest,"r_session.txt"))
periods <- read.csv(file.path(dest,"period_counts.csv"), colClasses=c(participant="character"))
trials <- read.csv(file.path(dest,"trial_metrics.csv"), colClasses=c(participant="character"))
results <- list()
diagnostics <- list()
jacobi <- matrix(0,40,40)
jacobi[cbind(1:39,2:40)] <- sqrt(1:39)
jacobi <- jacobi+t(jacobi)
quadrature <- eigen(jacobi,symmetric=TRUE)
z <- quadrature$values
w <- quadrature$vectors[1,]^2
stopifnot(abs(sum(w)-1)<1e-12,abs(sum(w*z))<1e-12,abs(sum(w*z*z)-1)<1e-12)

add_result <- function(variant,metric,test,family,method,estimate=NA_real_,ci_low=NA_real_,ci_high=NA_real_,statistic=NA_real_,df2=NA_real_,p_raw=NA_real_,...) {
  results[[length(results)+1L]] <<- data.frame(variant,metric,test,family,method,estimate,ci_low,ci_high,statistic,df2,p_raw,...)
}

er_margins <- function(fit) {
  grid <- expand.grid(condition=factor(c("C2","C3"),levels=c("C2","C3")),late=c(0,1))
  X <- model.matrix(~condition*late,grid)
  eta <- as.vector(X%*%fixef(fit))
  variance <- rep(0,4)
  for (v in VarCorr(fit)) {
    if (colnames(v)[1]=="(Intercept)") variance <- variance+v[1,1]
    else variance <- variance+grid$late^2*v[1,1]
  }
  p <- vapply(seq_len(4),function(i)sum(w*plogis(eta[i]+sqrt(variance[i])*z)),0.0)
  c(early_C2=p[1],early_C3=p[2],late_C2=p[3],late_C3=p[4],
    early_difference=p[1]-p[2],late_difference=p[3]-p[4],
    change_C2=p[3]-p[1],change_C3=p[4]-p[2],change_difference=p[3]-p[1]-p[4]+p[2])*100
}

fit_er <- function(variant) {
  d <- periods[periods$variant==variant,]
  stopifnot(nrow(d)==40,length(unique(d$participant))==20,all(table(d$participant)==2),length(unique(d$variant))==1)
  d$condition <- factor(d$condition,levels=c("C2","C3"))
  d$participant <- factor(d$participant)
  fit <- glmer(cbind(errors,total-errors)~condition*late+(1|participant)+(0+late|participant),
               data=d,family=binomial,nAGQ=1,na.action=na.fail,
               control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=100000)))
  saveRDS(fit,file.path(dest,paste0("glmm_",variant,".rds")))
  writeLines(capture.output(summary(fit)),file.path(dest,paste0("glmm_",variant,".txt")))
  tm <- system.time(pilot <- bootMer(fit,er_margins,nsim=20,seed=20261113,use.u=FALSE,parallel="no"))
  estimate_time <- unname(tm["elapsed"])*1999/20
  cat("GLMM",variant,"20-replicate seconds",tm["elapsed"],"predicted full seconds",estimate_time,"\n");flush.console()
  if (estimate_time>600) stop("Bootstrap estimate exceeds ten minutes at one core; resource review required")
  cache <- file.path(dest,paste0("glmm_bootstrap_",variant,".rds"))
  bt <- if(file.exists(cache))readRDS(cache) else bootMer(fit,er_margins,nsim=1999,seed=20261113+as.integer(variant!="base"),use.u=FALSE,parallel="no")
  stopifnot(isTRUE(all.equal(unname(bt$t0),unname(er_margins(fit)),tolerance=1e-8)))
  saveRDS(bt,file.path(dest,paste0("glmm_bootstrap_",variant,".rds")))
  good <- apply(bt$t,1,function(x)all(is.finite(x)))
  if (sum(good)<1900) stop("More than 5% bootstrap failures; inference requires review")
  estimates <- er_margins(fit)
  se <- apply(bt$t[good,,drop=FALSE],2,sd)
  quant <- apply(bt$t[good,,drop=FALSE],2,quantile,probs=c(.025,.975))
  map <- c(early_difference="early_C2-C3",late_difference="late_C2-C3",change_C2="change_C2",change_C3="change_C3",change_difference="change_C2-C3")
  for (key in names(map)) {
    stat <- estimates[key]/se[key]
    add_result(variant,"ER",map[[key]],if(key=="early_difference")"exploratory" else "secondary",
               "binomial GLMM marginal contrast; parametric-bootstrap SE Wald",unname(estimates[key]),
               unname(estimates[key]-qnorm(.975)*se[key]),unname(estimates[key]+qnorm(.975)*se[key]),
               unname(stat),NA_real_,unname(2*pnorm(-abs(stat))),percentile_low=quant[1,key],percentile_high=quant[2,key])
  }
  write.csv(data.frame(quantity=names(estimates),estimate=estimates,se=se,percentile_low=quant[1,],percentile_high=quant[2,]),file.path(dest,paste0("glmm_margins_",variant,".csv")),row.names=FALSE)
  diagnostics[[paste0("glmm_",variant)]] <<- list(singular=isSingular(fit),convergence=fit@optinfo$conv,
       random_variance=lapply(VarCorr(fit),unclass),pearson_dispersion=sum(residuals(fit,type="pearson")^2)/df.residual(fit),
       bootstrap_valid=sum(good),bootstrap_fail=attr(bt,"bootFail"),bootstrap_messages=lapply(attr(bt,"boot.all.msgs"),function(x)as.list(setNames(as.integer(x),names(x)))),
       predicted_bootstrap_seconds=estimate_time,logit_interaction=unname(fixef(fit)["conditionC3:late"]))
  cat("GLMM finished",variant,"\n");flush.console()
}

lag_correlation <- function(res,d) {
  adjacent <- which(as.character(d$participant[-1])==as.character(d$participant[-nrow(d)]) &
                      d$block[-1]==d$block[-nrow(d)] & diff(d$N)==1)+1
  if(length(adjacent)<3)return(0)
  cor(res[adjacent],res[adjacent-1])
}

fit_gamm <- function(variant,metric) {
  d <- trials[trials$variant==variant,]
  stopifnot(nrow(d)==if(variant=="base")7983 else 7154,length(unique(d$variant))==1,!anyDuplicated(d[c("participant","N")]))
  d <- d[order(d$participant,d$N),]
  d$condition <- factor(d$condition,levels=c("C2","C3"))
  d$ordered_condition <- ordered(d$condition,levels=c("C2","C3"))
  d$participant <- factor(d$participant)
  d$block <- factor(d$block)
  d$y <- d[[metric]]
  d$start <- c(TRUE,as.character(d$participant[-1])!=as.character(d$participant[-nrow(d)]) |
                       d$block[-1]!=d$block[-nrow(d)] | diff(d$N)!=1)
  fam <- if(metric=="ER")binomial() else gaussian()
  formula <- y~condition*block+s(N,k=10,bs="cr")+s(N,by=ordered_condition,k=10,bs="cr")+s(N,participant,bs="fs",k=5)
  fit0 <- bam(formula,data=d,family=fam,method="fREML",discrete=TRUE,nthreads=1,na.action=na.fail,control=gam.control(maxit=1000))
  rho <- max(-.95,min(.95,lag_correlation(residuals(fit0,type="pearson"),d)))
  fit <- bam(formula,data=d,family=fam,method="fREML",discrete=TRUE,nthreads=1,rho=rho,AR.start=d$start,na.action=na.fail,control=gam.control(maxit=1000))
  saveRDS(fit,file.path(dest,paste0("gamm_",variant,"_",metric,".rds")))
  writeLines(capture.output(summary(fit)),file.path(dest,paste0("gamm_",variant,"_",metric,".txt")))
  grid <- expand.grid(N=1:400,condition=factor(c("C2","C3"),levels=c("C2","C3")))
  grid$ordered_condition <- ordered(grid$condition,levels=c("C2","C3"))
  grid$block <- factor((grid$N-1)%/%100+1,levels=levels(d$block))
  grid$participant <- factor(levels(d$participant)[1],levels=levels(d$participant))
  X <- predict(fit,grid,type="lpmatrix",exclude="s(N,participant)")
  beta <- coef(fit)
  V <- vcov(fit,unconditional=TRUE)
  D <- X[1:400,,drop=FALSE]-X[401:800,,drop=FALSE]
  H <- sweep(D,2,colMeans(D))
  delta <- as.vector(H%*%beta)
  covariance <- H%*%V%*%t(H)
  eigen_shape <- eigen(covariance,symmetric=TRUE)
  keep <- eigen_shape$values>max(eigen_shape$values)*1e-8
  stat <- sum(as.vector(crossprod(eigen_shape$vectors[,keep,drop=FALSE],delta))^2/eigen_shape$values[keep])
  add_result(variant,metric,"curve_shape","exploratory","GAMM centered-contrast joint approximate penalized Wald",statistic=stat,df2=sum(keep),p_raw=pchisq(stat,sum(keep),lower.tail=FALSE))
  eigen_v <- eigen(V,symmetric=TRUE)
  L <- sweep(eigen_v$vectors,2,sqrt(pmax(eigen_v$values,0)),"*")
  set.seed(20261113+match(metric,c("MT","TPe","ER"))+10*as.integer(variant!="base"))
  draws <- L%*%matrix(rnorm(ncol(X)*10000),nrow=ncol(X))
  se_shape <- sqrt(pmax(diag(covariance),0))
  crit <- unname(quantile(apply(abs((H%*%draws)/pmax(se_shape,1e-12)),2,max),.95))
  link <- as.vector(X%*%beta)
  fit_se <- sqrt(pmax(rowSums((X%*%V)*X),0))
  response <- fam$linkinv(link)
  response_low <- fam$linkinv(link-1.96*fit_se)
  response_high <- fam$linkinv(link+1.96*fit_se)
  scale <- if(metric=="ER")100 else 1
  output <- data.frame(N=1:400,C2=response[1:400]*scale,C3=response[401:800]*scale,
     C2_low=response_low[1:400]*scale,C2_high=response_high[1:400]*scale,
     C3_low=response_low[401:800]*scale,C3_high=response_high[401:800]*scale,
     centered_difference=delta,simultaneous_low=delta-crit*se_shape,simultaneous_high=delta+crit*se_shape)
  write.csv(output,file.path(dest,paste0("gamm_curve_",variant,"_",metric,".csv")),row.names=FALSE)
  if(metric=="ER") {
    for(window in c("early","late")) {
      a <- if(window=="early")1 else 301
      b <- if(window=="early")100 else 400
      ids <- c(b,a,b+400,a+400)
      signs <- c(1,-1,-1,1)
      estimate <- sum(response[ids]*signs)*100/99
      gradient <- colSums(X[ids,,drop=FALSE]*(signs*response[ids]*(1-response[ids])))*100/99
      se <- sqrt(as.numeric(t(gradient)%*%V%*%gradient))
      stat <- estimate/se
      add_result(variant,metric,paste0(window,"_slope_C2-C3"),"exploratory","binomial GAMM mean probability slope contrast; delta Wald",estimate,estimate-1.96*se,estimate+1.96*se,stat,NA_real_,2*pnorm(-abs(stat)))
    }
  }
  set.seed(20261113)
  check <- k.check(fit,subsample=5000,n.rep=200)
  write.csv(check,file.path(dest,paste0("gamm_kcheck_",variant,"_",metric,".csv")))
  diagnostics[[paste0("gamm_",variant,"_",metric)]] <<- list(converged=fit$converged && fit$iter<1000,iterations=fit$iter,initial_converged=fit0$converged && fit0$iter<1000,initial_iterations=fit0$iter,rho=rho,rows=nrow(d),
      raw_residual_lag1=lag_correlation(residuals(fit,type="pearson"),d),
      whitened_lag1=if(!is.null(fit$std.rsd))lag_correlation(fit$std.rsd,d) else NA,
      residual_sd_by_condition=tapply(residuals(fit,type="pearson"),d$condition,sd),
      edf=sum(fit$edf),shape_rank=sum(keep),band_critical=crit,kcheck=unclass(check),
      note=if(metric=="ER")"binomial working-residual AR is a GEE approximation; probability curves set participant effects to zero" else "Gaussian AR(1); common residual variance assumption")
  cat("GAMM finished",variant,metric,"rho",rho,"\n");flush.console()
}

if(length(args)>1 && args[2]=="verify") {
  participant <- read.csv(file.path(dest,"participant_summary.csv"),colClasses=c(participant="character"))
  reported <- read.csv(file.path(dest,"all_tests.csv"))
  checked <- 0L
  for(variant in c("base","sensitivity")) {
    for(metric in c("MT","TPe")) {
      d <- participant[participant$variant==variant & participant$metric==metric,]
      long <- data.frame(participant=factor(rep(d$participant,2)),condition=factor(rep(d$condition,2)),
                         period=factor(rep(c("early","late"),each=nrow(d))),value=c(d$early,d$late))
      aov_result <- summary(aov(value~condition*period+Error(participant/period),data=long))
      interaction <- aov_result[["Error: participant:period"]][[1]]["condition:period",]
      target <- reported[reported$variant==variant & reported$metric==metric & reported$test=="change_C2-C3",]
      stopifnot(abs(interaction[["F value"]]-target$F)<1e-9,abs(interaction[["Pr(>F)"]]-target$p_raw)<1e-9)
      late <- summary(aov(late~condition,data=d))[[1]]["condition",]
      target <- reported[reported$variant==variant & reported$metric==metric & reported$test=="late_C2-C3",]
      stopifnot(abs(late[["F value"]]-target$F)<1e-9,abs(late[["Pr(>F)"]]-target$p_raw)<1e-9)
      checked <- checked+2L
    }
    fit <- readRDS(file.path(dest,paste0("glmm_",variant,".rds")))
    grid <- expand.grid(condition=factor(c("C2","C3"),levels=c("C2","C3")),late=c(0,1))
    eta <- as.vector(model.matrix(~condition*late,grid)%*%fixef(fit))
    variance <- vapply(VarCorr(fit),function(x)x[1,1],0.0)
    direct <- vapply(1:4,function(i)integrate(function(z)plogis(eta[i]+sqrt(variance[1]+grid$late[i]^2*variance[2])*z)*dnorm(z),-Inf,Inf,rel.tol=1e-10)$value,0.0)*100
    stopifnot(max(abs(direct-unname(er_margins(fit)[1:4])))<1e-7)
    for(family in c("primary","secondary","exploratory")) {
      d <- reported[reported$variant==variant & reported$family==family,]
      stopifnot(max(abs(p.adjust(d$p_raw,method="holm")-d$p_adjusted),na.rm=TRUE)<1e-12)
    }
  }
  write_json(list(anova_contrasts_checked=checked,quadrature_vs_adaptive_integration=TRUE,holm_vs_R=TRUE),
             file.path(dest,"cross_language_checks.json"),pretty=TRUE,auto_unbox=TRUE)
  cat("Cross-language ANOVA, marginal ER integration, and Holm checks passed\n")
  quit(save="no")
}

for (variant in c("base","sensitivity")) {
  fit_er(variant)
  for(metric in c("MT","TPe","ER"))fit_gamm(variant,metric)
  columns <- unique(unlist(lapply(results,names)))
  table <- do.call(rbind,lapply(results,function(r){for(k in setdiff(columns,names(r)))r[[k]]<-NA;r[,columns]}))
  write.csv(table,file.path(dest,"model_tests.csv"),row.names=FALSE,na="")
  write_json(diagnostics,file.path(dest,"model_diagnostics.json"),pretty=TRUE,auto_unbox=TRUE,na="null",digits=16)
}
writeLines(capture.output(warnings()),file.path(dest,"r_warnings.txt"))
